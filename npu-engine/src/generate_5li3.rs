//! Real, resident-pool (5li3 schedule) autoregressive generate loop: tokenizer
//! -> CPU prefill (`forward.rs`) -> per-token NPU decode step (`decode.rs`) ->
//! CPU lm_head (`forward::Model::logits`) -> sampler (`sampler.rs`) ->
//! detokenize. Ported from `tools/kernel-interp/generate_npu.py` (the decode
//! loop + `serve_config()`) and `tools/kernel-interp/run_5li3_npu.py` (the
//! prompt -> prefill-state buffer-prep half).
//!
//! Unlike `generate_npu.py`'s subprocess + per-token file hand-off, this
//! drives [`crate::decode::Driver`] in-process via
//! [`crate::decode::Driver::step_bytes`] (see `decode.rs`'s `load_resident`),
//! with zero per-token file I/O — only the one-time prefill-state write below
//! touches disk.
//!
//! # Known limitation (mirrors the Python handoff note, but narrower)
//!
//! `tools/server/backends.py`'s `NpuBackend` flagged that arbitrary per-request
//! prefill wasn't wired — this file fixes exactly that (every `generate()`
//! call reprefills the given `prompt_ids` on CPU and writes fresh
//! `state_L{l}.bin` files before booting the driver). What it does *not* do is
//! keep one resident driver alive across multiple `generate()` calls on the
//! same `Li3Backend`: because the 5li3 config's `buf state{l} ... <file>`
//! directive only loads its initial content at buffer-creation time (see
//! `decode.rs`'s `exec_line`/`load_resident`), a new prompt's state can only
//! reach the device by re-running `load_resident` against freshly-written
//! state files — i.e. by rebuilding the resident driver. So each call to
//! `generate()` reprefills, rewrites state, and reopens the device fresh; only
//! the `Model` (the ~GB-scale `.q4nx` mmap + layer schedule) is cached across
//! calls. This is honest, working behavior — it is what `generate_npu.py`
//! itself does (one driver process per invocation) — just not a
//! multi-request-resident driver, which would need a way to reload a live
//! buffer's content that `Driver` doesn't expose today.

use crate::forward::LayerState;
use crate::state_io;
use std::path::{Path, PathBuf};

/// Parameters for the 5li3 (5-layer, interval-3) resident schedule: model
/// file, xclbin dir, kernel elf dir, and the output dir holding the
/// prebuilt-and-reusable (prompt-independent) `pool_L*.bin`/`pack_L*.bin`/
/// `side_L*.bin`/`pool_lmhead.bin` files plus the per-prompt `state_L*.bin`
/// files this module (re)writes.
#[derive(Clone, Debug)]
pub struct Li3Config {
    pub model_path: PathBuf,
    pub xclbin_dir: PathBuf,
    pub kernel_dir: PathBuf,
    pub output_dir: PathBuf,
    pub num_layers: usize,
}

impl Default for Li3Config {
    fn default() -> Li3Config {
        Li3Config {
            model_path: PathBuf::from(
                "C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model_5Li3.q4nx",
            ),
            xclbin_dir: PathBuf::from("C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"),
            kernel_dir: PathBuf::from("C:/caps/m0c"),
            output_dir: PathBuf::from("C:/code/FastFlowLM/npu-engine/m3out/5li3"),
            num_layers: 5,
        }
    }
}

fn to_fwd(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// Build the resident-mode driver config text: device/xclbin/kernel lines,
/// `buf poolN/packN/sideN/stateN` per layer, `buf act`/`buf logits`, `serve`,
/// the runlist-chunks-of-<=3-with-cross-context-barrier layer program,
/// `endserve`. Mirrors `generate_npu.py::serve_config()` /
/// `tools/server/backends.py::NpuBackend._write_serve_config` byte-for-byte
/// in structure (generalized over `cfg.num_layers`).
pub fn build_serve_config_text(cfg: &Li3Config) -> String {
    let xb = to_fwd(&cfg.xclbin_dir);
    let cap = to_fwd(&cfg.kernel_dir);
    let d = to_fwd(&cfg.output_dir);
    let n = cfg.num_layers;

    let mut lines: Vec<String> = vec![
        "device".to_string(),
        format!("xclbin L {xb}/layer.xclbin"),
        format!("xclbin LM {xb}/lm_head.xclbin"),
        format!("kernel k0 L {cap}/elf_000005.bin"),
        format!("kernel k1 L {cap}/elf_000006.bin"),
        format!("kernel klm LM {cap}/elf_000003.bin"),
    ];
    for l in 0..n {
        lines.push(format!("buf pool{l} 536870912 {d}/pool_L{l}.bin"));
        lines.push(format!("buf pack{l} 2097152 {d}/pack_L{l}.bin"));
        lines.push(format!("buf side{l} 6291456 {d}/side_L{l}.bin"));
        lines.push(format!("buf state{l} 3145728 {d}/state_L{l}.bin")); // initial = fresh prefill state
    }
    lines.push(format!("buf lmpool 542113792 {d}/pool_lmhead.bin"));
    lines.push("buf act 1048576".to_string());
    lines.push("buf logits 1048576".to_string());
    lines.push("serve".to_string());

    // Matches generate_npu.py's kernel choice exactly: layer 0 uses k0, every
    // other layer uses k1 (a pool-layout quirk of the built xclbin/elfs, not a
    // function of layer type — ported as-is, not re-derived).
    let kern = |l: usize| if l == 0 { "k0" } else { "k1" };
    let mut chunk: Vec<usize> = Vec::new();
    for l in 0..n {
        chunk.push(l);
        if chunk.len() == 3 {
            lines.push("runlist L".to_string());
            for &c in &chunk {
                lines.push(format!("layer {} pool{c} act pack{c} side{c} state{c}", kern(c)));
            }
            lines.push("submit".to_string());
            lines.push("barrier klm logits lmpool act".to_string());
            chunk.clear();
        }
    }
    if !chunk.is_empty() {
        lines.push("runlist L".to_string());
        for &c in &chunk {
            lines.push(format!("layer {} pool{c} act pack{c} side{c} state{c}", kern(c)));
        }
        lines.push("submit".to_string());
    }
    // Trailing barrier: reset the layer.xclbin context before the next step's
    // chunk (decode.rs's module doc: keeps the <=3-consecutive-submit cap
    // satisfied across step boundaries, not just within one step).
    lines.push("barrier klm logits lmpool act".to_string());
    lines.push("endserve".to_string());
    lines.join("\n") + "\n"
}

/// Write [`build_serve_config_text`] to `{output_dir}/gen_serve.txt` and
/// return its path.
pub fn write_serve_config(cfg: &Li3Config) -> Result<PathBuf, String> {
    std::fs::create_dir_all(&cfg.output_dir)
        .map_err(|e| format!("create {}: {e}", cfg.output_dir.display()))?;
    let text = build_serve_config_text(cfg);
    let path = cfg.output_dir.join("gen_serve.txt");
    std::fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

/// Serialize each layer's CPU-computed prefill state into the driver's 3MB
/// binary state format and write `{output_dir}/state_L{l}.bin` — the Rust
/// port of `run_5li3_npu.py`'s buffer-prep role (`serialize_linear_state`/
/// `serialize_kv_state` + the `state_L{l}.bin` write), so the resident config
/// built by [`write_serve_config`] loads THIS prompt's prefill state, not a
/// fixed offline one.
pub fn write_states(cfg: &Li3Config, states: &[LayerState]) -> Result<(), String> {
    std::fs::create_dir_all(&cfg.output_dir)
        .map_err(|e| format!("create {}: {e}", cfg.output_dir.display()))?;
    for (l, st) in states.iter().enumerate() {
        let bytes = match st {
            LayerState::Linear(ls) => state_io::serialize_linear_state(ls),
            LayerState::Full(kv) => state_io::serialize_kv_state(kv),
        };
        let path = cfg.output_dir.join(format!("state_L{l}.bin"));
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Everything below touches the real NPU device (via decode.rs/xrt.rs) and is
// compiled only with `--features npu`, mirroring how main.rs gates `decode`/
// `xrt` — with `npu` off, `open-qwen-npu`'s plain `cargo build` still compiles
// this file cleanly (the config/state helpers above have no XRT dependency).
// ---------------------------------------------------------------------------
#[cfg(feature = "npu")]
mod npu_impl {
    use super::{write_serve_config, write_states, Li3Config};
    use crate::backend::{Backend, GenParams};
    use crate::decode;
    use crate::forward::{self, Model, HIDDEN};
    use crate::npu_lock::NpuLock;
    use crate::sampler::Sampler;
    use crate::state_io;
    use crate::tokenizer;

    /// Resident-pool (5li3) NPU generate backend. Holds the (expensive to
    /// open) `.q4nx` model mmap across calls; everything device-side is
    /// (re)built fresh inside each `generate()` call — see the module-level
    /// doc comment's "Known limitation" for why.
    pub struct Li3Backend {
        cfg: Li3Config,
        model: Option<Model>,
    }

    impl Li3Backend {
        pub fn new(cfg: Li3Config) -> Li3Backend {
            Li3Backend { cfg, model: None }
        }
    }

    impl Backend for Li3Backend {
        fn generate(
            &mut self,
            prompt_ids: &[u32],
            params: &mut GenParams,
            on_token: &mut dyn FnMut(u32),
        ) -> Result<(), String> {
            if params.max_tokens == 0 {
                return Ok(());
            }
            if self.model.is_none() {
                self.model = Some(forward::open_model(&self.cfg.model_path));
            }
            let model = self.model.as_ref().unwrap();
            let norm_weight = model.file.bf16("model.norm.weight");
            let stop_ids = params.stop_ids.clone();
            let is_stop = |id: u32| stop_ids.contains(&id) || tokenizer::EOS_TOKEN_IDS.contains(&id);

            let _lock = NpuLock::acquire("li3")?;

            // ---- CPU prefill (this prompt, not a fixed offline one) --------
            let ids_i64: Vec<i64> = prompt_ids.iter().map(|&t| t as i64).collect();
            eprintln!("li3: prefilling {} prompt tokens (CPU)...", ids_i64.len());
            let (x, states) = forward::run_prefill(model, &ids_i64);
            write_states(&self.cfg, &states)?;
            let t = ids_i64.len();
            let last_residual = x[(t - 1) * HIDDEN..t * HIDDEN].to_vec();

            let mut history = ids_i64.clone();

            // ---- first generated token, from the prefill's last-position ---
            // residual (position T-1 predicts token T) — mirrors
            // generate_npu.py::first_token_logits.
            let lg0 = model.logits(&last_residual);
            if !lg0.iter().all(|v| v.is_finite()) {
                return Err("non-finite prefill logits (interval-3 blowup?)".to_string());
            }
            let first = params.sampler.sample(&lg0, &history) as u32;

            // ---- boot the resident driver with THIS prompt's fresh state ---
            let cfg_path = write_serve_config(&self.cfg)?;
            let (mut driver, prog) = decode::load_resident(&cfg_path)?;
            eprintln!("li3: driver resident (pools + states loaded)");

            let mut emitted = 0usize;
            if is_stop(first) {
                return Ok(());
            }
            on_token(first);
            history.push(first as i64);
            let mut cur = first;
            emitted += 1;

            // ---- NPU decode loop --------------------------------------------
            while emitted < params.max_tokens {
                let embed = model.embed(cur as usize);
                let act = state_io::write_act(&embed, &norm_weight);
                let hidden = driver.step_bytes(&prog, &act)?;
                let hidden_f64: Vec<f64> = hidden[..HIDDEN * 2]
                    .chunks_exact(2)
                    .map(|c| crate::q4nx::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
                    .collect();
                if !hidden_f64.iter().all(|v| v.is_finite()) {
                    return Err("non-finite hidden (interval-3 blowup?)".to_string());
                }
                let lg = model.logits(&hidden_f64);
                let nxt = params.sampler.sample(&lg, &history) as u32;
                if is_stop(nxt) {
                    break;
                }
                on_token(nxt);
                history.push(nxt as i64);
                cur = nxt;
                emitted += 1;
            }
            Ok(())
        }
    }
}

#[cfg(feature = "npu")]
pub use npu_impl::Li3Backend;

#[cfg(all(test, feature = "npu"))]
mod tests {
    use super::*;
    use crate::backend::{Backend, GenParams};
    use crate::sampler::Sampler;
    use crate::tokenizer::{ChatMessage, Tokenizer, EOS_TOKEN_IDS};
    use std::collections::HashSet;

    /// End-to-end, real-hardware smoke test: tokenizer + chat template ->
    /// Li3Backend::generate -> detokenize. Requires the local model dir,
    /// prebuilt m3out/5li3 pool/pack/side/lm_head files, and the real NPU
    /// (exclusive-locked via NpuLock so a parallel workstream on the same
    /// machine doesn't collide). Run with:
    ///   cargo test --release --features npu --bin open-qwen-npu \
    ///     generate_5li3:: -- --ignored --nocapture
    #[test]
    #[ignore]
    fn generates_a_few_tokens_on_real_hardware() {
        let cfg = Li3Config::default();
        let model_dir = cfg.model_path.parent().expect("model dir").to_path_buf();
        let tok = Tokenizer::load(&model_dir).expect("load tokenizer");

        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "What is the capital of France?".to_string(),
        }];
        let prompt = tok
            .apply_chat_template(&msgs, true, false)
            .expect("render chat template");
        let ids = tok.encode(&prompt).expect("encode prompt");
        println!("prompt ({} tokens): {:?}", ids.len(), prompt);

        let sampler = Sampler::new(0.0, 0, 1.0, 1.0, None); // greedy, deterministic
        let mut stop_ids: HashSet<u32> = HashSet::new();
        stop_ids.extend(EOS_TOKEN_IDS.iter().copied());
        let mut params = GenParams { max_tokens: 8, sampler, stop_ids };

        let mut backend = Li3Backend::new(cfg);
        let mut out_ids: Vec<u32> = Vec::new();
        backend
            .generate(&ids, &mut params, &mut |t| out_ids.push(t))
            .expect("generate");

        assert!(!out_ids.is_empty(), "no tokens were generated");
        assert!(
            out_ids.iter().all(|&id| (id as usize) < crate::forward::VOCAB),
            "generated an out-of-vocab token id"
        );
        let text = tok.decode(&out_ids).expect("decode generated ids");

        println!("generated {} tokens: {:?}", out_ids.len(), out_ids);
        println!("decoded:\n{text}");

        assert!(!text.is_empty(), "decoded text was empty");
    }
}
