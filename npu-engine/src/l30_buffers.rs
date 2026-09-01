//! Build all NPU buffers for a real per-request prefill of the full 30-layer
//! streamed model, ported from `tools/kernel-interp/l30_build.py`.
//!
//! Almost all of `l30_build.py`'s heavy lifting (CPU prefill math, pool/pack/
//! side byte layout) is already ported and hardware-verified elsewhere —
//! `forward::run_prefill` does the CPU math, `pools::build_layer_pool` /
//! `build_pack` / `build_side` / `build_lmhead_pool` do the byte layout, and
//! `state_io::serialize_linear_state` / `serialize_kv_state` do the state
//! serialization (same format as `l30_build.py`'s own, already round-trip
//! tested against `main.rs`'s read path). This file is just the
//! orchestration `l30_build.py`'s `__main__` block does: loop over layers,
//! write files to an output dir, compute the first decode token.
//!
//! No NPU/XRT dependency at all — this is pure CPU math + file I/O, so unlike
//! `decode.rs`/`xrt.rs` it is NOT behind the `npu` feature and always builds.

use crate::forward::{LayerState, Model, HIDDEN};
use crate::pools;
use crate::sampler;
use crate::state_io;
use std::path::{Path, PathBuf};

/// Everything [`generate_l30`](crate) needs from a request's buffer build:
/// where the buffers landed, the layer schedule (for the streaming config
/// generator), and the first decode token (computed here on the CPU from the
/// prefill's final hidden state, same as `l30_build.py`'s
/// `first = int(logits.argmax())`).
pub struct L30Build {
    pub out_dir: PathBuf,
    /// "linear_attention" / "full_attention" per layer, in layer order.
    pub layer_types: Vec<String>,
    pub first_token: u32,
}

fn write_file(path: &Path, data: &[u8]) -> Result<(), String> {
    std::fs::write(path, data).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Run CPU prefill over `ids` (via `forward::run_prefill`) and write every
/// NPU buffer the streamed 30-layer decode schedule needs into `out_dir`:
/// `pool_L{l}.bin` (512MB), `pack_L{l}.bin` (2MB), `side_L{l}.bin` (6MB), and
/// `state_L{l}.bin` (3MB) per layer — always rewritten, since (unlike the
/// weight pools) the per-layer *state* is prefill-dependent, i.e. specific to
/// this request's prompt. The lm_head pool (`pool_lmhead.bin`, ~540MB) is the
/// one buffer that's truly prompt-independent (and model-wide, not even
/// per-layer) — like `pools_only_l40.py`, skip rebuilding it if it's already
/// on disk from an earlier request against the same model.
pub fn build(model: &Model, ids: &[i64], out_dir: &Path) -> Result<L30Build, String> {
    if ids.is_empty() {
        return Err("l30_buffers::build: empty prompt".to_string());
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
        drop(pool); // ~512MB; drop before building the next layer's pool

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

    Ok(L30Build { out_dir: out_dir.to_path_buf(), layer_types: model.layer_types.clone(), first_token })
}
