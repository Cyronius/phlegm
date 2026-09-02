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
    /// The instruction bytes as loaded (before any `moeroute` patching).
    instr_src: HashMap<String, Vec<u8>>,
    /// `moeroute` patch table per kernel: (word index, expert slot 0..8, core, kind 0/1/2).
    moe_patches: HashMap<String, Vec<(usize, usize, u64, u8)>>,
    /// name -> (bo, logical size). The Bo's own size is padded up to 1 MB.
    bufs: HashMap<String, (Bo, usize)>,
    layeridx: i32,
}

// moe_experts (tools/open-kernels/designs/moe_experts/moe_experts.py) weight
// stream layout: per expert [up 4 stripes | gate 4 stripes | down 16 bands],
// core c reads up/gate stripe c (128 rows) and down bands 2c, 2c+1 (256 rows).
const MOE_STRIPE: u64 = 163_840;
const MOE_UP_BYTES: u64 = 4 * MOE_STRIPE;
const MOE_EXPERT_BYTES: u64 = 3 * MOE_UP_BYTES;
const MOE_DOWN_CORE: u64 = 81_920;
// Layer pool regions (pools.rs): routed gate/up stripes interleaved
// [up0 gate0 up1 gate1 ...] from 0, down experts, shared up/gate/down.
const MOE_POOL_DOWN: u64 = 335_544_320;
const MOE_POOL_SHARE_UP: u64 = 503_316_480;
const MOE_POOL_SHARE_GATE: u64 = 503_971_840;
const MOE_POOL_SHARE_DOWN: u64 = 504_627_200;

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
            instr_src: HashMap::new(),
            moe_patches: HashMap::new(),
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

    /// The `moeroute` patch table for a moe_experts kernel, built once from the
    /// instruction stream as loaded: every DDR-patch op (0x81, 12 words:
    /// reg addr at +6, arg index at +8, byte offset at +10) on arg 0 is one
    /// weight fill whose static offset into the host-built `wexp` names its
    /// expert slot, core and kind.
    fn moe_patch_table(&mut self, kn: &str) -> Result<Vec<(usize, usize, u64, u8)>, String> {
        if let Some(t) = self.moe_patches.get(kn) {
            return Ok(t.clone());
        }
        let src = self.instr_src.get(kn).ok_or(format!("no kernelx {kn}"))?;
        let w: Vec<u32> = src.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut t = Vec::new();
        let mut i = 4;
        while i < w.len() {
            let len = match w[i] {
                0 => 6,
                1 => 12,
                3 => 7,
                0x80 => 4,
                0x81 => 12,
                _ => 1,
            };
            if w[i] == 0x81 && i + 11 < w.len() && w[i + 8] == 0 {
                let off = w[i + 10] as u64;
                let slot = (off / MOE_EXPERT_BYTES) as usize;
                let rem = off % MOE_EXPERT_BYTES;
                let (kind, core) = if rem < MOE_UP_BYTES {
                    (0u8, rem / MOE_STRIPE)
                } else if rem < 2 * MOE_UP_BYTES {
                    (1u8, (rem - MOE_UP_BYTES) / MOE_STRIPE)
                } else {
                    (2u8, (rem - 2 * MOE_UP_BYTES) / MOE_DOWN_CORE)
                };
                t.push((i + 10, slot, core, kind));
            }
            i += len;
        }
        if t.len() != 144 {
            return Err(format!("moeroute: {kn} has {} weight fills, expected 144", t.len()));
        }
        self.moe_patches.insert(kn.to_string(), t.clone());
        Ok(t)
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
                self.instr_src.insert(name.clone(), insts);
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
                let t0 = std::time::Instant::now();
                r.execute()?;
                r.wait().map_err(|e| format!("RUNLIST FAILED: {e}"))?;
                println!(
                    "runlist[{} runs] -> completed ({:.3} ms)",
                    rl_runs.len(),
                    t0.elapsed().as_secs_f64() * 1e3
                );
                self.layeridx += rl_runs.len() as i32;
                *rl = None;
                rl_runs.clear();
            }
            "copy" => {
                // copy <dst> <dst_off> <src> <src_off> <nbytes>: BO -> BO through
                // host memory (a host round trip; assembles one kernel's input
                // from other kernels' outputs until the fused designs write it
                // in place).
                let dst = it.next().ok_or("copy: dst")?.to_string();
                let doff: usize = it.next().ok_or("copy: dst_off")?.parse().map_err(|_| "copy: bad dst_off")?;
                let src = it.next().ok_or("copy: src")?.to_string();
                let soff: usize = it.next().ok_or("copy: src_off")?.parse().map_err(|_| "copy: bad src_off")?;
                let n: usize = it.next().ok_or("copy: nbytes")?.parse().map_err(|_| "copy: bad nbytes")?;
                let mut tmp = vec![0u8; n];
                {
                    let s = self.bufs.get(&src).ok_or(format!("no buf {src}"))?;
                    s.0.sync_from_device()?;
                    s.0.read(&mut tmp, soff)?;
                }
                let d = self.bufs.get_mut(&dst).ok_or(format!("no buf {dst}"))?;
                d.0.write(&tmp, doff)?;
                d.0.sync_to_device()?;
            }
            "moeroute" => {
                // moeroute <kernel> <rout-buf>: point the moe_experts kernel's
                // routed-expert fills at the experts the router just chose.
                // The kernel's instruction stream carries one DDR-patch op per
                // fill (arg 0 = the weight BO, byte offset into it); built for
                // a host-concatenated `wexp`, those offsets identify (slot,
                // core, up/gate/down), and are rewritten here as offsets into
                // the resident layer pool (pools.rs layout). ~0.1 ms of host
                // time per layer instead of a 15 MB host slice.
                let kn = it.next().ok_or("moeroute: kernel")?.to_string();
                let rb = it.next().ok_or("moeroute: rout buf")?;
                let t0 = std::time::Instant::now();
                let mut idx = [0u8; 32];
                {
                    let r = self.bufs.get(rb).ok_or(format!("no buf {rb}"))?;
                    r.0.sync_from_device()?;
                    r.0.read(&mut idx, 1024)?; // router output: int32 idx[8] at byte 1024
                }
                let idx: Vec<u32> = idx.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                let table = self.moe_patch_table(&kn)?;
                let (ibo, _) = self.instr.get_mut(&kn).ok_or(format!("no kernelx {kn}"))?;
                for &(word, slot, core, kind) in &table {
                    let off: u64 = if slot < 8 {
                        let e = idx[slot] as u64;
                        match kind {
                            0 => (8 * e + 2 * core) * MOE_STRIPE,
                            1 => (8 * e + 2 * core + 1) * MOE_STRIPE,
                            _ => MOE_POOL_DOWN + e * MOE_UP_BYTES + core * MOE_DOWN_CORE,
                        }
                    } else {
                        match kind {
                            0 => MOE_POOL_SHARE_UP + core * MOE_STRIPE,
                            1 => MOE_POOL_SHARE_GATE + core * MOE_STRIPE,
                            _ => MOE_POOL_SHARE_DOWN + core * MOE_DOWN_CORE,
                        }
                    };
                    ibo.write(&(off as u32).to_le_bytes(), word * 4)?;
                }
                ibo.sync_to_device()?;
                println!("moeroute {kn} idx {:?} ({:.3} ms)", idx, t0.elapsed().as_secs_f64() * 1e3);
            }
            "runx" => {
                // `run`'s arg binding, but queued on the open `runlist` instead
                // of submitted alone (all runs of one runlist share its xclbin
                // context). Measures what a same-context sequence costs
                // without the per-run submit/wait floor.
                let kn = it.next().ok_or("runx: kernel")?.to_string();
                let names: Vec<String> = it.map(|s| s.to_string()).collect();
                let args: Vec<(i32, &str)> =
                    names.iter().enumerate().map(|(i, n)| (3 + i as i32, n.as_str())).collect();
                let r = self.make_run(&kn, &args)?;
                rl_runs.push(r);
                rl.as_mut().ok_or("runx without runlist")?.add(rl_runs.last().unwrap())?;
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
            "ctrlpkt" => {
                // Build AIE control-packet words that retarget a shim DMA BD at a
                // slab of another buffer and push it to a task queue:
                //   ctrlpkt <dst-buf> <target-buf> <bd_reg> <queue_reg> <bd_id> <slab_bytes> <idx> [addr_bias]
                // Packet word layout (mlir-aie AIETargetNPU.cpp / add_one_ctrl_packet):
                //   header = stream_id<<24 | opcode<<22 | (beats=n-1)<<20 | addr, parity in bit 31
                // followed by `n` data words. opcode 0 = write.
                let dst = it.next().ok_or("ctrlpkt: dst buf")?.to_string();
                let tgt = it.next().ok_or("ctrlpkt: target buf")?.to_string();
                let pnum = |s: Option<&str>, what: &str| -> Result<u64, String> {
                    let s = s.ok_or(format!("ctrlpkt: {what}"))?;
                    let s2 = s.trim_start_matches("0x");
                    if s2.len() != s.len() { u64::from_str_radix(s2, 16) } else { s.parse::<u64>() }
                        .map_err(|_| format!("ctrlpkt: bad {what}: {s}"))
                };
                let bd_reg = pnum(it.next(), "bd_reg")? as u32;
                let q_reg = pnum(it.next(), "queue_reg")? as u32;
                let bd_id = pnum(it.next(), "bd_id")? as u32;
                let slab = pnum(it.next(), "slab_bytes")?;
                let idx = pnum(it.next(), "idx")?;
                let bias = it.next().map(|s| {
                    let s2 = s.trim_start_matches("0x");
                    if s2.len() != s.len() { u64::from_str_radix(s2, 16).unwrap_or(0) } else { s.parse().unwrap_or(0) }
                }).unwrap_or(0);

                let base = self.bufs.get(&tgt).ok_or(format!("no buf {tgt}"))?.0.address();
                let addr = base.wrapping_add(bias).wrapping_add(idx * slab);
                let parity = |n: u32| -> u32 { if (n.count_ones() % 2) == 0 { 1 } else { 0 } };
                let hdr = |addr_reg: u32, n: u32| -> u32 {
                    let h = ((n - 1) << 20) | (addr_reg & 0xFFFFF); // stream_id 0, opcode 0 (write)
                    h | (parity(h) << 31)
                };
                let mut words: Vec<u32> = Vec::new();
                // slab_bytes == 0 => enqueue only, leaving whatever address the
                // firmware already patched into the BD. That isolates the
                // control-packet/enqueue mechanism from the address encoding.
                if slab != 0 {
                    // BD word1 = base_address_low (bits [1:0] are zero), word2 = high 16 bits.
                    words.push(hdr(bd_reg + 4, 2));
                    words.push((addr & 0xFFFF_FFFC) as u32);
                    words.push(((addr >> 32) & 0xFFFF) as u32);
                }
                // Push the BD to its task queue: enable_token_issue | start_bd_id,
                // the same value the compiler and FLM write.
                words.push(hdr(q_reg, 1));
                words.push(0x8000_0000 | bd_id);
                println!("ctrlpkt {dst}: base={base:#x} bias={bias:#x} idx={idx} -> addr={addr:#x}, {} words", words.len());
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let entry = self.bufs.get_mut(&dst).ok_or(format!("no buf {dst}"))?;
                entry.0.init(&[])?;
                entry.0.write(&bytes, 0)?;
                entry.0.sync_to_device()?;
            }
            "setwords" => {
                // setwords <buf> <word-offset> <v> [<v> ...]  -- write u32 words (dec or 0x hex)
                let name = it.next().ok_or("setwords: buf")?.to_string();
                let off: usize = it.next().ok_or("setwords: offset")?.parse().map_err(|_| "setwords: bad offset")?;
                let vals: Vec<u32> = it
                    .map(|t| {
                        let t2 = t.trim_start_matches("0x");
                        if t2.len() != t.len() { u32::from_str_radix(t2, 16).unwrap_or(0) } else { t.parse().unwrap_or(0) }
                    })
                    .collect();
                let bytes: Vec<u8> = vals.iter().flat_map(|w| w.to_le_bytes()).collect();
                let entry = self.bufs.get_mut(&name).ok_or(format!("no buf {name}"))?;
                entry.0.write(&bytes, off * 4)?;
                entry.0.sync_to_device()?;
            }
            "boaddr" => {
                // boaddr <buf> <word-offset> <target-buf> [bias]  -- write target's DDR
                // address (+bias) as two u32 words (lo, hi) into buf at word-offset.
                let name = it.next().ok_or("boaddr: buf")?.to_string();
                let off: usize = it.next().ok_or("boaddr: offset")?.parse().map_err(|_| "boaddr: bad offset")?;
                let tgt = it.next().ok_or("boaddr: target")?.to_string();
                let bias: u64 = it.next().map(|t| {
                    let t2 = t.trim_start_matches("0x");
                    if t2.len() != t.len() { u64::from_str_radix(t2, 16).unwrap_or(0) } else { t.parse().unwrap_or(0) }
                }).unwrap_or(0);
                let addr = self.bufs.get(&tgt).ok_or(format!("no buf {tgt}"))?.0.address().wrapping_add(bias);
                let bytes: Vec<u8> = [(addr & 0xFFFF_FFFF) as u32, (addr >> 32) as u32].iter().flat_map(|w| w.to_le_bytes()).collect();
                println!("boaddr {name}[{off}] = {addr:#x} ({tgt}+{bias:#x})");
                let entry = self.bufs.get_mut(&name).ok_or(format!("no buf {name}"))?;
                entry.0.write(&bytes, off * 4)?;
                entry.0.sync_to_device()?;
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
