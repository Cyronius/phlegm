//! Rust port of `npu-engine/m0/decode_driver.cpp`: a single-process NPU decode
//! driver over the `xrt-shim` FFI ([`crate::xrt`]).
//!
//! It interprets the same config directive language as the C++ driver, so the
//! exact same config files (e.g. `m3out/5li3/drv_5li3_rl.txt`) drive it. The
//! decode-critical shapes are preserved verbatim:
//!
//! - Kernels (elf -> module -> ext::kernel) are created ONCE and each decode
//!   invocation just re-binds args on a fresh `xrt::run` and re-submits, exactly
//!   as FLM's engine does. Creating/leaking a run per invocation causes resource
//!   accumulation and a timeout after a few layers.
//! - The layer.xclbin hw_context times out after ~3 consecutive submissions, so
//!   decode batches layers into `runlist` chunks of <=3 and inserts a
//!   cross-context lm_head submission ("barrier") between chunks to reset the
//!   queue. Keep the <=3-chunk + cross-context-barrier pattern.
//! - `serve` keeps pools AND per-layer states resident across tokens; per token
//!   it loads the `act` buffer, replays the fixed layer program, and dumps the
//!   hidden state, with a trailing barrier each step to keep the layer queue
//!   under the ~3 cap across step boundaries.

use crate::xrt::{Bo, Context, Device, Kernel, Runlist, STATE_COMPLETED};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

/// Interpreter state, mirroring decode_driver.cpp's locals.
pub struct Driver {
    dev: Option<Device>,
    ctxs: HashMap<String, Context>,
    kernels: HashMap<String, Kernel>,
    /// Classic-flow kernels (`kernelx`, mlir-aie xclbin + insts.bin): their
    /// instruction-stream BO and its 32-bit word count, bound at args 1/2 of
    /// every run. ELF-flow kernels carry instructions in the module (args 1/2 = 0).
    instr: HashMap<String, (Bo, usize)>,
    /// name -> (bo, logical size). The Bo's own size is padded up to 1 MB.
    bufs: HashMap<String, (Bo, usize)>,
    layeridx: i32,
}

fn read_file(p: &str) -> Result<Vec<u8>, String> {
    std::fs::read(p).map_err(|e| format!("cannot open {p}: {e}"))
}

impl Driver {
    fn new() -> Driver {
        Driver {
            dev: None,
            ctxs: HashMap::new(),
            kernels: HashMap::new(),
            instr: HashMap::new(),
            bufs: HashMap::new(),
            layeridx: 0,
        }
    }

    fn dev(&self) -> &Device {
        self.dev.as_ref().expect("device not opened")
    }

    /// r.set_arg(0,3),(1,0),(2,0) then data BOs at the given arg indices.
    fn make_run(
        &self,
        kn: &str,
        args: &[(i32, &str)],
    ) -> Result<crate::xrt::Run, String> {
        let k: &Kernel = self.kernels.get(kn).ok_or(format!("no kernel {kn}"))?;
        let mut r = k.run()?;
        r.set_arg_int(0, 3)?;
        if let Some((ibo, nwords)) = self.instr.get(kn) {
            r.set_arg_bo(1, ibo)?;
            r.set_arg_int(2, *nwords as i32)?;
        } else {
            r.set_arg_int(1, 0)?;
            r.set_arg_int(2, 0)?;
        }
        for (idx, bufname) in args {
            let bo: &Bo = &self
                .bufs
                .get(*bufname)
                .ok_or(format!("no buf {bufname}"))?
                .0;
            r.set_arg_bo(*idx, bo)?;
        }
        Ok(r)
    }

    /// Single-shot submit (its own runlist implied) for lm_head / standalone ops.
    fn submit_single(&self, kn: &str, args: &[(i32, &str)]) -> Result<i32, String> {
        let mut r = self.make_run(kn, args)?;
        r.start()?;
        r.wait()
    }

    /// Run one config/program line. `rl`/`rl_runs` carry the pending runlist.
    /// Returns Ok(true) to keep going, Ok(false) to stop (unused today).
    fn exec_line(
        &mut self,
        line: &str,
        rl: &mut Option<Runlist>,
        rl_runs: &mut Vec<crate::xrt::Run>,
    ) -> Result<(), String> {
        let mut it = line.split_whitespace();
        let cmd = match it.next() {
            Some(c) => c,
            None => return Ok(()),
        };
        match cmd {
            "device" => {
                let d = Device::open(0)?;
                println!("device: {}", d.name());
                self.dev = Some(d);
            }
            "xclbin" => {
                let name = it.next().ok_or("xclbin: name")?.to_string();
                let path = it.next().ok_or("xclbin: path")?;
                let ctx = self.dev().hwctx(Path::new(path))?;
                self.ctxs.insert(name.clone(), ctx);
                println!("xclbin {name}");
            }
            "kernel" => {
                let name = it.next().ok_or("kernel: name")?.to_string();
                let xn = it.next().ok_or("kernel: xclbin")?;
                let elfp = it.next().ok_or("kernel: elf")?;
                let ctx = self.ctxs.get(xn).ok_or(format!("no xclbin {xn}"))?;
                let k = ctx.kernel(Path::new(elfp))?;
                self.kernels.insert(name.clone(), k);
                println!("kernel {name} ({elfp})");
            }
            "kernelx" => {
                // Classic mlir-aie flow: `kernelx <name> <xclbin-ctx> <insts.bin>`
                // -> xrt::kernel(ctx, "MLIR_AIE") + a cacheable instruction BO.
                let name = it.next().ok_or("kernelx: name")?.to_string();
                let xn = it.next().ok_or("kernelx: xclbin")?;
                let instp = it.next().ok_or("kernelx: insts.bin")?;
                let ctx = self.ctxs.get(xn).ok_or(format!("no xclbin {xn}"))?;
                let k = ctx.kernel_xclbin("MLIR_AIE")?;
                let insts = read_file(instp)?;
                let mut ibo = self.dev().bo_instr(&k, insts.len())?;
                ibo.write(&insts, 0)?;
                ibo.sync_to_device()?;
                println!("kernelx {name} ({instp}, {} words)", insts.len() / 4);
                self.instr.insert(name.clone(), (ibo, insts.len() / 4));
                self.kernels.insert(name, k);
            }
            "buf" => {
                let name = it.next().ok_or("buf: name")?.to_string();
                let size: usize = it.next().ok_or("buf: size")?.parse().map_err(|_| "buf: bad size")?;
                let initf = it.next();
                let mut bo = self.dev().bo(size)?;
                if let Some(f) = initf {
                    let d = read_file(f)?;
                    bo.init(&d)?;
                } else {
                    bo.init(&[])?;
                }
                bo.sync_to_device()?;
                self.bufs.insert(name, (bo, size));
            }
            "load" => {
                let name = it.next().ok_or("load: name")?;
                let initf = it.next().ok_or("load: file")?;
                let d = read_file(initf)?;
                let entry = self.bufs.get_mut(name).ok_or(format!("no buf {name}"))?;
                let n = d.len().min(entry.1);
                entry.0.write(&d[..n], 0)?;
                entry.0.sync_to_device()?;
            }
            "runlist" => {
                let xn = it.next().ok_or("runlist: xclbin")?;
                let ctx = self.ctxs.get(xn).ok_or(format!("no xclbin {xn}"))?;
                *rl = Some(ctx.runlist()?);
                rl_runs.clear();
            }
            "layer" => {
                let kn = it.next().ok_or("layer: kernel")?.to_string();
                let pool = it.next().ok_or("layer: pool")?.to_string();
                let act = it.next().ok_or("layer: act")?.to_string();
                let pack = it.next().ok_or("layer: pack")?.to_string();
                let side = it.next().ok_or("layer: side")?.to_string();
                let state = it.next().ok_or("layer: state")?.to_string();
                let args = [
                    (3, pool.as_str()),
                    (4, act.as_str()),
                    (5, pack.as_str()),
                    (6, side.as_str()),
                    (7, state.as_str()),
                ];
                if rl.is_some() {
                    let r = self.make_run(&kn, &args)?;
                    rl_runs.push(r);
                    rl.as_mut().unwrap().add(rl_runs.last().unwrap())?;
                } else {
                    // immediate single submit
                    rl_runs.clear();
                    let mut r = self.make_run(&kn, &args)?;
                    r.start()?;
                    let st = r.wait()?;
                    println!("layer[{}] {kn} -> state {st}", self.layeridx);
                    self.layeridx += 1;
                    if st != STATE_COMPLETED {
                        return Err("LAYER FAILED".to_string());
                    }
                }
            }
            "submit" => {
                let r = rl.as_mut().ok_or("submit without runlist")?;
                r.execute()?;
                r.wait().map_err(|e| format!("RUNLIST FAILED: {e}"))?;
                println!("runlist[{} layers] -> completed", rl_runs.len());
                self.layeridx += rl_runs.len() as i32;
                *rl = None;
                rl_runs.clear();
            }
            "lmhead" => {
                let kn = it.next().ok_or("lmhead: kernel")?.to_string();
                let logits = it.next().ok_or("lmhead: logits")?.to_string();
                let lmpool = it.next().ok_or("lmhead: lmpool")?.to_string();
                let act = it.next().ok_or("lmhead: act")?.to_string();
                let st = self.submit_single(
                    &kn,
                    &[(3, logits.as_str()), (4, lmpool.as_str()), (5, act.as_str())],
                )?;
                println!("lmhead {kn} -> state {st}");
                if st != STATE_COMPLETED {
                    return Err("LMHEAD FAILED".to_string());
                }
            }
            "run" => {
                // Generic immediate submit: `run <kernel> <buf> [<buf> ...]` binds
                // the buffers at args 3.. (after opcode/instr/ninstr) and waits.
                // For open (IRON-built) designs whose arg list isn't the fused
                // layer / lm_head shape.
                let kn = it.next().ok_or("run: kernel")?.to_string();
                let names: Vec<String> = it.map(|s| s.to_string()).collect();
                if names.is_empty() {
                    return Err("run: needs at least one buffer".to_string());
                }
                let args: Vec<(i32, &str)> =
                    names.iter().enumerate().map(|(i, n)| (3 + i as i32, n.as_str())).collect();
                let t0 = std::time::Instant::now();
                let st = self.submit_single(&kn, &args)?;
                println!("run {kn} [{} bufs] -> state {st} ({:.3} ms)", names.len(), t0.elapsed().as_secs_f64() * 1e3);
                if st != STATE_COMPLETED {
                    return Err("RUN FAILED".to_string());
                }
            }
            "dump" => {
                let name = it.next().ok_or("dump: name")?;
                let outf = it.next().ok_or("dump: file")?;
                let size: usize = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                let entry = self.bufs.get(name).ok_or(format!("no buf {name}"))?;
                let n = if size != 0 { size } else { entry.1 };
                entry.0.sync_from_device()?;
                let mut v = vec![0u8; n];
                entry.0.read(&mut v, 0)?;
                std::fs::write(outf, &v).map_err(|e| format!("write {outf}: {e}"))?;
            }
            "loglogits" => {
                let name = it.next().ok_or("loglogits: name")?;
                let entry = self.bufs.get(name).ok_or(format!("no buf {name}"))?;
                entry.0.sync_from_device()?;
                let n = 124160usize;
                let mut raw = vec![0u8; n * 4];
                entry.0.read(&mut raw, 0)?;
                let lg: Vec<f32> = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let mut finite = true;
                let mut amax = 0f32;
                let mut arg = 0usize;
                for i in 0..n {
                    if !lg[i].is_finite() {
                        finite = false;
                    }
                    if lg[i].is_finite() && lg[i].abs() > amax {
                        amax = lg[i].abs();
                    }
                    if lg[i].is_finite() && lg[i] > lg[arg] {
                        arg = i;
                    }
                }
                println!(
                    "logits: finite={} absmax={:.3} argmax_vocab={}",
                    finite as i32,
                    amax,
                    2 * arg + 1
                );
            }
            "serve" => {
                return Err("serve must be handled by run_config".to_string());
            }
            other => {
                return Err(format!("bad directive: {other}"));
            }
        }
        Ok(())
    }

    /// One resident decode step, entirely in-process: write `act_bytes` into
    /// the resident `act` buffer, run the fixed per-token layer program
    /// (`prog`, the lines between `serve`/`endserve`), and return the
    /// resulting 8192-byte hidden dump. No subprocess, no temp files — this is
    /// what [`serve`](Driver::serve) is built on, and what a real generate
    /// loop should call directly instead of spawning this binary as a
    /// subprocess and shuttling activations through disk.
    ///
    /// `prog` may also contain `load <buf> <file>` lines (same as the
    /// immediate-mode `load` directive in [`exec_line`](Driver::exec_line)):
    /// this is what a pool-STREAMING per-token program needs — e.g. the l30
    /// schedule's 30 layers can't all stay resident as 512MB pools, so its
    /// program reloads a small set of pool buffers from disk before each
    /// group of layers, exactly as `l30_run_npu.py`'s `gen_stream_cfg` does.
    /// A resident schedule (e.g. 5li3) simply never emits a `load` line.
    pub fn step_bytes(&mut self, prog: &[String], act_bytes: &[u8]) -> Result<[u8; 8192], String> {
        {
            let entry = self.bufs.get_mut("act").ok_or("no buf act")?;
            let n = act_bytes.len().min(entry.1);
            entry.0.write(&act_bytes[..n], 0)?;
            entry.0.sync_to_device()?;
        }
        // run the fixed program (runlist chunks + cross-context barriers)
        let mut prl: Option<Runlist> = None;
        let mut prl_runs: Vec<crate::xrt::Run> = Vec::new();
        for p in prog {
            let mut ls = p.split_whitespace();
            let c = match ls.next() {
                Some(c) => c,
                None => continue,
            };
            match c {
                "load" => {
                    let name = ls.next().ok_or("load: name")?.to_string();
                    let initf = ls.next().ok_or("load: file")?;
                    let d = read_file(initf)?;
                    let entry = self.bufs.get_mut(&name).ok_or(format!("no buf {name}"))?;
                    let n = d.len().min(entry.1);
                    entry.0.write(&d[..n], 0)?;
                    entry.0.sync_to_device()?;
                }
                "runlist" => {
                    let xn = ls.next().ok_or("runlist: xclbin")?;
                    let ctx = self.ctxs.get(xn).ok_or(format!("no xclbin {xn}"))?;
                    prl = Some(ctx.runlist()?);
                    prl_runs.clear();
                }
                "layer" => {
                    let kn = ls.next().ok_or("layer: kernel")?.to_string();
                    let pool = ls.next().ok_or("layer: pool")?.to_string();
                    let act = ls.next().ok_or("layer: act")?.to_string();
                    let pack = ls.next().ok_or("layer: pack")?.to_string();
                    let side = ls.next().ok_or("layer: side")?.to_string();
                    let state = ls.next().ok_or("layer: state")?.to_string();
                    let args = [
                        (3, pool.as_str()),
                        (4, act.as_str()),
                        (5, pack.as_str()),
                        (6, side.as_str()),
                        (7, state.as_str()),
                    ];
                    let r = self.make_run(&kn, &args)?;
                    prl_runs.push(r);
                    prl.as_mut().ok_or("layer without runlist")?.add(prl_runs.last().unwrap())?;
                }
                "submit" => {
                    let r = prl.as_mut().ok_or("submit without runlist")?;
                    r.execute()?;
                    r.wait().map_err(|e| format!("STEP FAILED: {e}"))?;
                    prl = None;
                    prl_runs.clear();
                }
                "barrier" => {
                    let kn = ls.next().ok_or("barrier: kernel")?.to_string();
                    let logits = ls.next().ok_or("barrier: logits")?.to_string();
                    let lmpool = ls.next().ok_or("barrier: lmpool")?.to_string();
                    let act = ls.next().ok_or("barrier: act")?.to_string();
                    self.submit_single(&kn, &[(3, logits.as_str()), (4, lmpool.as_str()), (5, act.as_str())])?;
                }
                _ => {}
            }
        }
        // dump hidden (act, first 8192 bytes)
        let entry = self.bufs.get("act").ok_or("no buf act")?;
        entry.0.sync_from_device()?;
        let mut hidden = [0u8; 8192];
        entry.0.read(&mut hidden, 0)?;
        Ok(hidden)
    }

    /// Resident decode loop over stdin (subprocess mode, as spawned by
    /// generate_npu.py / tools/server's NpuBackend today). Reads lines:
    ///   step <act_in> <hidden_out>   -> load act, run program, dump 8192 B hidden
    ///   quit                          -> exit
    /// Thin file-I/O wrapper around [`step_bytes`](Driver::step_bytes); kept
    /// for the standalone `npu <config>` CLI subcommand and for driving this
    /// binary as a subprocess from another process/language.
    fn serve(&mut self, prog: &[String]) -> Result<(), String> {
        println!("SERVE READY");
        std::io::stdout().flush().ok();
        let stdin = std::io::stdin();
        for req in stdin.lock().lines() {
            let req = req.map_err(|e| format!("stdin: {e}"))?;
            let mut rs = req.split_whitespace();
            let op = match rs.next() {
                Some(o) => o,
                None => continue,
            };
            if op == "quit" {
                break;
            }
            if op != "step" {
                continue;
            }
            let actin = rs.next().ok_or("step: act_in")?.to_string();
            let hidout = rs.next().ok_or("step: hidden_out")?.to_string();
            let act_bytes = read_file(&actin)?;
            match self.step_bytes(prog, &act_bytes) {
                Ok(hidden) => {
                    std::fs::write(&hidout, hidden).map_err(|e| format!("write {hidout}: {e}"))?;
                    println!("STEP OK");
                }
                Err(e) => println!("STEP ERR: {e}"),
            }
            std::io::stdout().flush().ok();
        }
        println!("SERVE DONE");
        Ok(())
    }
}

/// Parse a decode-driver config file and run every directive up to (but not
/// including) `serve`, leaving kernels/contexts/buffers resident. Returns the
/// live `Driver` plus the fixed per-token layer program (the lines between
/// `serve` and `endserve`) — this is the in-process counterpart to
/// [`run_config`]'s subprocess-oriented parsing, for a real generate loop to
/// drive directly via repeated [`Driver::step_bytes`] calls with no
/// subprocess and no per-token file I/O.
pub fn load_resident(cfg_path: &Path) -> Result<(Driver, Vec<String>), String> {
    let text = std::fs::read_to_string(cfg_path)
        .map_err(|e| format!("cannot open config {}: {e}", cfg_path.display()))?;
    let lines: Vec<&str> = text.lines().collect();

    let mut drv = Driver::new();
    let mut rl: Option<Runlist> = None;
    let mut rl_runs: Vec<crate::xrt::Run> = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim_end();
        i += 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.split_whitespace().next() == Some("serve") {
            let mut prog = Vec::new();
            while i < lines.len() {
                let pl = lines[i].trim_end().to_string();
                i += 1;
                if pl == "endserve" {
                    break;
                }
                if !pl.is_empty() && !pl.starts_with('#') {
                    prog.push(pl);
                }
            }
            return Ok((drv, prog));
        }
        if let Err(e) = drv.exec_line(line, &mut rl, &mut rl_runs) {
            return Err(format!("on '{line}': {e}"));
        }
    }
    Err("config has no `serve` block".to_string())
}

/// Interpret a decode-driver config file.
pub fn run_config(cfg_path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(cfg_path)
        .map_err(|e| format!("cannot open config {}: {e}", cfg_path.display()))?;
    let lines: Vec<&str> = text.lines().collect();

    let mut drv = Driver::new();
    let mut rl: Option<Runlist> = None;
    let mut rl_runs: Vec<crate::xrt::Run> = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim_end();
        i += 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // `serve` swallows the remaining lines up to `endserve` as its program.
        if line.split_whitespace().next() == Some("serve") {
            let mut prog = Vec::new();
            while i < lines.len() {
                let pl = lines[i].trim_end().to_string();
                i += 1;
                if pl == "endserve" {
                    break;
                }
                if !pl.is_empty() && !pl.starts_with('#') {
                    prog.push(pl);
                }
            }
            drv.serve(&prog)?;
            continue;
        }
        if let Err(e) = drv.exec_line(line, &mut rl, &mut rl_runs) {
            return Err(format!("on '{line}': {e}"));
        }
    }
    println!("DONE");
    Ok(())
}
