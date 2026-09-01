//! Build the NPU buffers for the full 30-layer streamed model, ported from
//! `tools/kernel-interp/l30_build.py`.
//!
//! Most of `l30_build.py`'s heavy lifting (pool/pack/side byte layout) is
//! already ported and hardware-verified elsewhere — `pools::build_layer_pool`
//! / `build_pack` / `build_side` / `build_lmhead_pool` do the byte layout,
//! and `state_io::serialize_linear_state` / `serialize_kv_state` do the state
//! serialization (used only by [`build_cpu_prefill_reference`] below now).
//!
//! No NPU/XRT dependency at all — this is pure CPU math + file I/O, so unlike
//! `decode.rs`/`xrt.rs` it is NOT behind the `npu` feature and always builds.

use crate::forward::{LayerState, Model, HIDDEN};
use crate::pools;
use crate::sampler;
use crate::state_io;
use std::path::{Path, PathBuf};

/// Everything [`generate_l30`](crate) needs from a request's buffer build:
/// where the buffers landed and the layer schedule (for the streaming config
/// generator). No CPU-computed first-decode-token any more — the caller
/// determines it from the NPU's own final prefill hidden state via
/// decode-as-prefill (same mechanism `L40Backend` already ships), not from
/// anything computed here.
pub struct L30Build {
    pub out_dir: PathBuf,
    /// "linear_attention" / "full_attention" per layer, in layer order.
    pub layer_types: Vec<String>,
}

const STATE_BYTES: usize = 3_145_728;

fn write_file(path: &Path, data: &[u8]) -> Result<(), String> {
    std::fs::write(path, data).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Write every prompt-INDEPENDENT NPU buffer the streamed 30-layer decode
/// schedule needs into `out_dir`: `pool_L{l}.bin` (512MB weights),
/// `pack_L{l}.bin` (2MB), `side_L{l}.bin` (6MB), and a ZERO-filled
/// `state_L{l}.bin` (3MB) per layer, plus the model-wide `pool_lmhead.bin`
/// (skipped if already present, like the other prompt-independent buffers).
///
/// Per-layer state used to be computed here from a CPU forward pass over the
/// prompt (see [`build_cpu_prefill_reference`]) — but that CPU model
/// (`forward::run_prefill`) is KNOWN DIVERGENT from verified NPU behavior
/// (0.57-0.68 corr, see the `flm-capture-oracle` memory / `.claude/plans/
/// l30-npu-prefill.md`), so real generation must not depend on it. Instead
/// every layer's state starts at zero, and the caller
/// (`L30Backend::generate`) runs the prompt through the streamed decode
/// kernel one token at a time ("decode-as-prefill" — mathematically exact
/// prefill because the layer kernel self-tracks its own KV/state position
/// on-device from zeroed state; the same mechanism `L40Backend` already ships
/// and verified against real FLM ground truth). This function no longer
/// needs the prompt token ids at all.
pub fn build(model: &Model, out_dir: &Path) -> Result<L30Build, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| format!("create_dir_all {}: {e}", out_dir.display()))?;

    let zero_state = vec![0u8; STATE_BYTES];
    for (l, lt) in model.layer_types.iter().enumerate() {
        let full = lt == "full_attention";

        let pool = pools::build_layer_pool(&model.file, l, full);
        write_file(&out_dir.join(format!("pool_L{l}.bin")), &pool)?;
        drop(pool); // ~512MB; drop before building the next layer's pool

        let pack = pools::build_pack(&model.file, l);
        write_file(&out_dir.join(format!("pack_L{l}.bin")), &pack)?;

        let side = pools::build_side(&model.file, l, full);
        write_file(&out_dir.join(format!("side_L{l}.bin")), &side)?;

        write_file(&out_dir.join(format!("state_L{l}.bin")), &zero_state)?;

        eprintln!("l30_buffers: wrote buffers L{l} ({lt})");
    }

    let lmhead_path = out_dir.join("pool_lmhead.bin");
    if lmhead_path.exists() {
        eprintln!("l30_buffers: pool_lmhead.bin already present, skipping (prompt-independent)");
    } else {
        let lm = pools::build_lmhead_pool(&model.file);
        write_file(&lmhead_path, &lm)?;
        eprintln!("l30_buffers: wrote pool_lmhead.bin");
    }

    Ok(L30Build { out_dir: out_dir.to_path_buf(), layer_types: model.layer_types.clone() })
}

/// The ORIGINAL CPU-prefill buffer build: runs `forward::run_prefill` (a full
/// from-scratch forward pass in Rust) over the prompt to compute each layer's
/// post-prompt state, and returns the CPU-computed first decode token
/// (greedy argmax) alongside the build.
///
/// NOT used by `L30Backend::generate` any more — `forward.rs` is KNOWN
/// DIVERGENT from verified NPU behavior (see the `flm-capture-oracle`
/// memory), so this must not drive real generation. Kept only as a
/// comparison point for verification layer 2 in
/// `.claude/plans/l30-npu-prefill.md`: diff these CPU-computed states against
/// the NPU decode-as-prefill states on the same prompt to catch gross bugs (a
/// wrong layer, a byte-order mistake) — agreement is not proof of
/// correctness, since this CPU path is the same compromised reference.
pub fn build_cpu_prefill_reference(model: &Model, ids: &[i64], out_dir: &Path) -> Result<(L30Build, u32), String> {
    if ids.is_empty() {
        return Err("l30_buffers::build_cpu_prefill_reference: empty prompt".to_string());
    }
    std::fs::create_dir_all(out_dir).map_err(|e| format!("create_dir_all {}: {e}", out_dir.display()))?;

    eprintln!("l30_buffers: CPU prefill over {} tokens, {} layers", ids.len(), model.layer_types.len());
    let (x, states) = crate::forward::run_prefill(model, ids);
    let t = ids.len();
    let x_last = &x[(t - 1) * HIDDEN..t * HIDDEN];
    let logits = model.logits(x_last);
    let first_token = sampler::greedy(&logits) as u32;
    eprintln!("l30_buffers: prefill done, first_token={first_token}");

    for (l, lt) in model.layer_types.iter().enumerate() {
        let full = lt == "full_attention";

        let pool = pools::build_layer_pool(&model.file, l, full);
        write_file(&out_dir.join(format!("pool_L{l}.bin")), &pool)?;
        drop(pool);

        let pack = pools::build_pack(&model.file, l);
        write_file(&out_dir.join(format!("pack_L{l}.bin")), &pack)?;

        let side = pools::build_side(&model.file, l, full);
        write_file(&out_dir.join(format!("side_L{l}.bin")), &side)?;

        let state_bytes = match &states[l] {
            LayerState::Linear(st) => state_io::serialize_linear_state(st),
            LayerState::Full(st) => state_io::serialize_kv_state(st),
        };
        write_file(&out_dir.join(format!("state_L{l}.bin")), &state_bytes)?;

        eprintln!("l30_buffers: wrote buffers L{l} ({lt})");
    }

    let lmhead_path = out_dir.join("pool_lmhead.bin");
    if lmhead_path.exists() {
        eprintln!("l30_buffers: pool_lmhead.bin already present, skipping (prompt-independent)");
    } else {
        let lm = pools::build_lmhead_pool(&model.file);
        write_file(&lmhead_path, &lm)?;
        eprintln!("l30_buffers: wrote pool_lmhead.bin");
    }

    Ok((L30Build { out_dir: out_dir.to_path_buf(), layer_types: model.layer_types.clone() }, first_token))
}
