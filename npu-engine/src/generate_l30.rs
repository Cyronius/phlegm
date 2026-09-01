//! The l30 (30-layer, pool-STREAMING) generate loop: `L30Backend`, a
//! `backend::Backend` for the schedule that can't keep its weight pools
//! resident (30 x 512MB pools = 15GB), unlike the 5li3 schedule.
//!
//! Ported from `tools/kernel-interp/l30_run_npu.py`'s `gen_stream_cfg` (config
//! generation) and `gen` mode (the decode loop), but restructured around the
//! in-process, no-subprocess driver: [`l30_buffers::build`] does the
//! per-request CPU prefill + buffer write (mirrors `l30_build.py`),
//! [`decode::load_resident`] opens the device/xclbins/kernels and the
//! resident buffers ONCE, and the decode loop calls
//! [`decode::Driver::step_bytes`] repeatedly instead of `l30_run_npu.py`'s
//! `run_driver` (which re-spawns `decode_driver.exe` as a fresh subprocess
//! for every single token — the very overhead this whole rewrite exists to
//! remove).
//!
//! Every streamed decode step still has to reload 30 x 512MB = 15GB off disk
//! (only 3 pool BOs — poolA/B/C — stay resident, reloaded 3-layers-at-a-time
//! via `load` lines in the per-token program), so this is legitimately slow
//! per token. That's inherent to the algorithm (pool streaming exists because
//! the pools can't all fit resident), not a bug in this port.

use crate::forward::{open_model, Model};
use crate::l30_buffers;
use std::path::{Path, PathBuf};

#[cfg(feature = "npu")]
use crate::backend::{Backend, GenParams};
#[cfg(feature = "npu")]
use crate::decode;
#[cfg(feature = "npu")]
use crate::q4nx::bf16_to_f32;
#[cfg(feature = "npu")]
use crate::sampler;
#[cfg(feature = "npu")]
use crate::state_io;

pub const DEFAULT_MODEL: &str = "C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model_30L.q4nx";
pub const DEFAULT_XCLBIN_DIR: &str = "C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2";
pub const DEFAULT_CAP_DIR: &str = "C:/caps/m0c";
pub const DEFAULT_OUT_DIR: &str = "C:/code/FastFlowLM/npu-engine/m3out/l30_gen";

/// A `Backend` for the streamed 30-layer schedule. Holds the open model (for
/// CPU prefill + CPU lm_head — the NPU only ever computes the raw per-layer
/// residual update, never the final logits) and the paths a request's config
/// text is built from. No NPU/XRT state is held between calls: each
/// `generate` call builds a fresh buffer set for its prompt, opens the
/// device, and tears it down when the call returns (dropping the driver and
/// releasing the NPU lock).
pub struct L30Backend {
    pub model: Model,
    pub out_dir: PathBuf,
    pub xclbin_dir: PathBuf,
    pub cap_dir: PathBuf,
}

impl L30Backend {
    pub fn new(
        model_path: &Path,
        out_dir: impl Into<PathBuf>,
        xclbin_dir: impl Into<PathBuf>,
        cap_dir: impl Into<PathBuf>,
    ) -> L30Backend {
        L30Backend { model: open_model(model_path), out_dir: out_dir.into(), xclbin_dir: xclbin_dir.into(), cap_dir: cap_dir.into() }
    }

    /// Convenience constructor using the default model/xclbin/elf/output
    /// paths (same directories `l30_run_npu.py` uses).
    pub fn open_default() -> L30Backend {
        L30Backend::new(Path::new(DEFAULT_MODEL), DEFAULT_OUT_DIR, DEFAULT_XCLBIN_DIR, DEFAULT_CAP_DIR)
    }
}

/// Build the streaming decode-driver config text for the l30 schedule:
/// header (device/xclbins/a single `kL` kernel — `l30_run_npu.py`'s `hdr()`
/// declares only `kL` from `elf_000005.bin`, unlike 5li3's k0/k1 split, since
/// every l30 layer uses the same kernel), then the buffer declarations
/// (`buf_decls` in the Python), then a `serve`/`endserve` block containing
/// the fixed per-token program: `NLAYERS/3` groups of
/// `load poolA/B/C -> runlist L -> layer x3 -> submit -> barrier`, mirroring
/// `gen_stream_cfg` line-for-line except:
///   - `act` has no init file (`buf act 1048576`, no trailing path) — it's
///     written per-token via `Driver::step_bytes`, not read from
///     `act_decode.bin` on disk like the Python version re-does every step.
///   - the per-token program's cross-context reset directive is named
///     `barrier` (`decode.rs`'s `Driver::step_bytes` program-directive name)
///     rather than `lmhead` (the immediate-mode directive name used for the
///     same op outside `serve`/`endserve` — see `decode.rs::exec_line` vs
///     `step_bytes`).
pub fn build_stream_config(out_dir: &Path, xclbin_dir: &Path, cap_dir: &Path, nlayers: usize) -> String {
    let out = out_dir.display();
    let xb = xclbin_dir.display();
    let cap = cap_dir.display();
    let mut s = String::new();
    s += "device\n";
    s += &format!("xclbin L {xb}/layer.xclbin\n");
    s += &format!("xclbin LM {xb}/lm_head.xclbin\n");
    s += &format!("kernel kL L {cap}/elf_000005.bin\n");
    s += &format!("kernel klm LM {cap}/elf_000003.bin\n");
    s += "buf poolA 536870912\n";
    s += "buf poolB 536870912\n";
    s += "buf poolC 536870912\n";
    s += &format!("buf lmpool 542113792 {out}/pool_lmhead.bin\n");
    s += "buf act 1048576\n";
    s += "buf logits 1048576\n";
    for l in 0..nlayers {
        s += &format!("buf pack{l} 2097152 {out}/pack_L{l}.bin\n");
    }
    for l in 0..nlayers {
        s += &format!("buf side{l} 6291456 {out}/side_L{l}.bin\n");
    }
    for l in 0..nlayers {
        s += &format!("buf state{l} 3145728 {out}/state_L{l}.bin\n");
    }
    s += "serve\n";
    let pool_names = ["poolA", "poolB", "poolC"];
    let mut base = 0;
    while base < nlayers {
        let group_len = (nlayers - base).min(3);
        for i in 0..group_len {
            s += &format!("load {} {out}/pool_L{}.bin\n", pool_names[i], base + i);
        }
        s += "runlist L\n";
        for i in 0..group_len {
            s += &format!("layer kL {} act pack{} side{} state{}\n", pool_names[i], base + i, base + i, base + i);
        }
        s += "submit\n";
        s += "barrier klm logits lmpool act\n";
        base += group_len;
    }
    s += "endserve\n";
    s
}

/// Parse the 8192-byte hidden dump `Driver::step_bytes` returns: the first
/// 4096 bytes are the raw (pre-final-norm) residual, bf16[2048] — the next
/// 4096 are `model.norm.weight` echoed back unchanged (kernel input format,
/// irrelevant on the way out). bf16 -> f32 -> f64 matches the 5li3 path and
/// what `forward::Model::logits` expects (it applies the final RMSNorm
/// itself).
#[cfg(feature = "npu")]
fn hidden_to_residual_f64(hidden: &[u8; 8192]) -> Vec<f64> {
    hidden[..4096]
        .chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
        .collect()
}

#[cfg(feature = "npu")]
use crate::npu_lock::NpuLock;

#[cfg(feature = "npu")]
impl Backend for L30Backend {
    /// Build this request's buffers (per-request real prefill, not a fixed
    /// offline one — closes the "known functional gap" noted in the rewrite
    /// plan), open the device once, then decode-step token by token,
    /// streaming 15GB off disk per step.
    fn generate(&mut self, prompt_ids: &[u32], params: &mut GenParams, on_token: &mut dyn FnMut(u32)) -> Result<(), String> {
        if prompt_ids.is_empty() {
            return Err("L30Backend::generate: empty prompt".to_string());
        }
        let ids: Vec<i64> = prompt_ids.iter().map(|&x| x as i64).collect();

        let build = l30_buffers::build(&self.model, &ids, &self.out_dir)?;

        let cfg_text = build_stream_config(&build.out_dir, &self.xclbin_dir, &self.cap_dir, build.layer_types.len());
        let cfg_path = build.out_dir.join("cfg_gen_stream.txt");
        std::fs::write(&cfg_path, &cfg_text).map_err(|e| format!("write {}: {e}", cfg_path.display()))?;

        // Exclusive NPU access for the lifetime of load_resident + all steps
        // below (device open through the last step_bytes call).
        let _lock = NpuLock::acquire("l30 gen")?;
        let (mut driver, prog) = decode::load_resident(&cfg_path)?;

        let norm_weight = self.model.file.bf16("model.norm.weight");
        let mut history: Vec<i64> = ids;

        let mut tok = build.first_token;
        for i in 0..params.max_tokens {
            if params.stop_ids.contains(&tok) {
                break;
            }
            on_token(tok);
            history.push(tok as i64);
            if i + 1 == params.max_tokens {
                break; // no need to run a decode step for a token we won't emit
            }

            let embed = self.model.embed(tok as usize);
            let act = state_io::write_act(&embed, &norm_weight);
            let hidden = driver.step_bytes(&prog, &act)?;
            let x_last = hidden_to_residual_f64(&hidden);
            let logits = self.model.logits(&x_last);
            tok = params.sampler.sample(&logits, &history) as u32;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Structure check against `l30_run_npu.py`'s `gen_stream_cfg`, no
    /// hardware/model needed: header, buf decls (poolA/B/C with no init
    /// file, lmpool + act + logits, then all packs/sides/states), then a
    /// serve block of `ceil(nlayers/3)` groups, each
    /// `load x{<=3} -> runlist L -> layer x{<=3} -> submit -> barrier`.
    #[test]
    fn stream_config_matches_python_gen_stream_cfg_structure() {
        let cfg = build_stream_config(Path::new("OUT"), Path::new("XB"), Path::new("CAP"), 6);
        let lines: Vec<&str> = cfg.lines().collect();
        let expected = vec![
            "device",
            "xclbin L XB/layer.xclbin",
            "xclbin LM XB/lm_head.xclbin",
            "kernel kL L CAP/elf_000005.bin",
            "kernel klm LM CAP/elf_000003.bin",
            "buf poolA 536870912",
            "buf poolB 536870912",
            "buf poolC 536870912",
            "buf lmpool 542113792 OUT/pool_lmhead.bin",
            "buf act 1048576",
            "buf logits 1048576",
            "buf pack0 2097152 OUT/pack_L0.bin",
            "buf pack1 2097152 OUT/pack_L1.bin",
            "buf pack2 2097152 OUT/pack_L2.bin",
            "buf pack3 2097152 OUT/pack_L3.bin",
            "buf pack4 2097152 OUT/pack_L4.bin",
            "buf pack5 2097152 OUT/pack_L5.bin",
            "buf side0 6291456 OUT/side_L0.bin",
            "buf side1 6291456 OUT/side_L1.bin",
            "buf side2 6291456 OUT/side_L2.bin",
            "buf side3 6291456 OUT/side_L3.bin",
            "buf side4 6291456 OUT/side_L4.bin",
            "buf side5 6291456 OUT/side_L5.bin",
            "buf state0 3145728 OUT/state_L0.bin",
            "buf state1 3145728 OUT/state_L1.bin",
            "buf state2 3145728 OUT/state_L2.bin",
            "buf state3 3145728 OUT/state_L3.bin",
            "buf state4 3145728 OUT/state_L4.bin",
            "buf state5 3145728 OUT/state_L5.bin",
            "serve",
            "load poolA OUT/pool_L0.bin",
            "load poolB OUT/pool_L1.bin",
            "load poolC OUT/pool_L2.bin",
            "runlist L",
            "layer kL poolA act pack0 side0 state0",
            "layer kL poolB act pack1 side1 state1",
            "layer kL poolC act pack2 side2 state2",
            "submit",
            "barrier klm logits lmpool act",
            "load poolA OUT/pool_L3.bin",
            "load poolB OUT/pool_L4.bin",
            "load poolC OUT/pool_L5.bin",
            "runlist L",
            "layer kL poolA act pack3 side3 state3",
            "layer kL poolB act pack4 side4 state4",
            "layer kL poolC act pack5 side5 state5",
            "submit",
            "barrier klm logits lmpool act",
            "endserve",
        ];
        assert_eq!(lines, expected);
    }

    /// For the real 30-layer model: `l30_buffers::build` writes every
    /// pool/pack/side/state file plus the lm_head pool, at the exact sizes
    /// the driver config declares. No NPU/hardware needed (pure CPU + disk),
    /// but writing ~15.75GB from a real model is minutes of wall-clock —
    /// `#[ignore]`d, invoke with a generous bounded timeout.
    #[test]
    #[ignore]
    fn l30_buffers_writes_all_files_with_correct_sizes() {
        let model_path = Path::new(DEFAULT_MODEL);
        let model = open_model(model_path);
        let nlayers = model.layer_types.len();
        assert!(nlayers > 0, "model has no layers");

        let model_dir = model_path.parent().unwrap();
        let tok = crate::tokenizer::Tokenizer::load(model_dir).expect("load tokenizer");
        let ids: Vec<i64> = tok.encode("The capital of France is").expect("encode").iter().map(|&x| x as i64).collect();

        let out_dir = Path::new("C:/code/FastFlowLM/npu-engine/m3out/l30_gen_bufcheck");
        let build = l30_buffers::build(&model, &ids, out_dir).expect("l30_buffers::build");

        assert_eq!(build.layer_types.len(), nlayers);
        assert_eq!(build.out_dir, out_dir);

        for l in 0..nlayers {
            let pool_len = std::fs::metadata(out_dir.join(format!("pool_L{l}.bin"))).unwrap().len();
            assert_eq!(pool_len, 536_870_912, "pool_L{l}.bin size");
            let pack_len = std::fs::metadata(out_dir.join(format!("pack_L{l}.bin"))).unwrap().len();
            assert_eq!(pack_len, 2_097_152, "pack_L{l}.bin size");
            let side_len = std::fs::metadata(out_dir.join(format!("side_L{l}.bin"))).unwrap().len();
            assert_eq!(side_len, 6_291_456, "side_L{l}.bin size");
            let state_len = std::fs::metadata(out_dir.join(format!("state_L{l}.bin"))).unwrap().len();
            assert_eq!(state_len, 3_145_728, "state_L{l}.bin size");
        }
        let lm_len = std::fs::metadata(out_dir.join("pool_lmhead.bin")).unwrap().len();
        assert_eq!(lm_len, 542_113_792, "pool_lmhead.bin size");

        println!("l30_buffers: wrote {nlayers} layers + lm_head pool to {} (first_token={})", out_dir.display(), build.first_token);
    }

    /// Real end-to-end: build buffers for a short prompt, open the device,
    /// stream through the full 30-layer schedule a handful of decode steps,
    /// sample real tokens. Requires local model + real hardware +
    /// `--features npu`; each step streams 15GB off disk so this is
    /// minutes-to-tens-of-minutes of wall clock — `#[ignore]`d, invoke with a
    /// generous bounded timeout.
    #[cfg(feature = "npu")]
    #[test]
    #[ignore]
    fn end_to_end_generate_a_few_tokens_on_npu() {
        let model_path = Path::new(DEFAULT_MODEL);
        let model_dir = model_path.parent().unwrap();
        let tok = crate::tokenizer::Tokenizer::load(model_dir).expect("load tokenizer");
        let prompt_ids = tok.encode("The capital of France is").expect("encode");

        let mut backend = L30Backend::open_default();
        let mut params = GenParams {
            max_tokens: 5,
            sampler: sampler::Sampler::new(0.0, 0, 1.0, 1.0, None), // greedy, deterministic
            stop_ids: std::collections::HashSet::new(),
        };
        let mut out: Vec<u32> = Vec::new();
        backend.generate(&prompt_ids, &mut params, &mut |t| out.push(t)).expect("generate");

        println!("generated token ids: {out:?}");
        if let Ok(text) = tok.decode(&out) {
            println!("generated text: {text:?}");
        }
        assert!(!out.is_empty(), "expected at least one generated token");
        assert!(out.len() <= params.max_tokens);
    }
}
