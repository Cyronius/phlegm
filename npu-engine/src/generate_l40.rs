//! The base-model (40-layer, resident-pool) generate loop: `L40Backend`, the
//! first `backend::Backend` with NO CPU forward math on the request path.
//!
//! Phase 1 of `docs/npu-prefill.md`: prompt tokens are prefilled by
//! SEQUENTIAL DECODE from zeroed states ("decode-as-prefill" — the decode
//! kernel self-tracks its KV position on-device, so feeding the prompt one
//! token at a time from zero states is mathematically exact prefill; verified
//! against FLM's own captured prefill boundary states, see the plan's
//! "Phase 0 VERIFIED" section). The final prompt token also runs the NPU
//! lm_head, whose output (full-vocab bf16, 496640 B) feeds the sampler
//! directly — `forward.rs` is not involved at all beyond embedding-row reads.
//!
//! Execution shape is the M5 `servep` pipeline, ported from
//! `m0/decode_driver_nobarrier.cpp`: two hw_contexts on the SAME layer.xclbin
//! ping-pong runlist chunks of <=3 layers (a submission to the other context
//! is what resets the per-context 3-consecutive-submit budget — no wasteful
//! lm_head barrier), and each chunk's `execute()` happens BEFORE the previous
//! chunk's `wait()` so completion overlaps submission.
//!
//! Unlike `Li3Backend` (which must rebuild its driver per request because its
//! prefill STATE arrives via buffer-creation-time file loads), this backend
//! keeps the device, kernels and all ~21 GB of pool BOs resident across
//! `generate()` calls: a new request only needs the 40 state BOs re-zeroed
//! (decode-as-prefill starts from zero by construction). First open is
//! minutes (pool upload); every later request pays nothing but its own
//! tokens. The NPU lockfile is held for as long as the backend is resident.

use std::path::PathBuf;

/// Paths for the 40-layer base-model schedule. `buf_dir` holds the prebuilt,
/// prompt-independent `pool_L*.bin` / `pack_L*.bin` / `side_L*.bin` /
/// `pool_lmhead.bin` (built by `tools/kernel-interp/pools_only_l40.py` /
/// `bench_e2e_l40.py`); `elf_dir` the captured kernel ELFs (same files every
/// backend uses).
#[derive(Clone, Debug)]
pub struct L40Config {
    pub model_path: PathBuf,
    pub xclbin_dir: PathBuf,
    pub elf_dir: PathBuf,
    pub buf_dir: PathBuf,
    pub num_layers: usize,
}

impl Default for L40Config {
    fn default() -> L40Config {
        // FLM_XCLBIN_DIR overrides where the closed .xclbin kernels are found
        // (they are NOT in this repo — see NOTICE.md / tools/get-kernels.ps1);
        // the fallback is a FastFlowLM checkout's copy.
        let xclbin_dir = std::env::var("FLM_XCLBIN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("C:/code/FastFlowLM/src/xclbins/Qwen3.6-35B-A3B-NPU2"));
        L40Config {
            model_path: PathBuf::from("C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx"),
            xclbin_dir,
            elf_dir: PathBuf::from("C:/caps/m0c"),
            buf_dir: PathBuf::from("C:/code/FastFlowLM/npu-engine/m3out/l40"),
            num_layers: 40,
        }
    }
}

#[cfg(feature = "npu")]
mod npu_impl {
    use super::L40Config;
    use crate::backend::{Backend, GenParams};
    use crate::npu_lock::NpuLock;
    use crate::q4nx::{bf16_to_f32, Q4nx};
    use crate::state_io;
    use crate::tokenizer;
    use crate::xrt::{Bo, Context, Device, Kernel, Run, Runlist, STATE_COMPLETED};
    use std::time::Instant;

    const HIDDEN: usize = 2048;
    /// lm_head output width (padded); real vocab rows below.
    const VOCAB: usize = 248_320;
    /// tokenizer.json's real vocab — logits above this are lm_head padding
    /// rows with undefined content, masked to -inf before sampling.
    const REAL_VOCAB: usize = 248_070;
    /// bytes of real logits in the 1 MB logits BO (248320 bf16).
    const LOGITS_BYTES: usize = VOCAB * 2;
    /// KV rows the 3 MB state buffer can hold (k bf16[T,512] in [0:1073152]);
    /// prompt+generation beyond this would overrun the pack region.
    const MAX_POSITIONS: usize = 1024;

    const POOL_BYTES: usize = 536_870_912;
    const PACK_BYTES: usize = 2_097_152;
    const SIDE_BYTES: usize = 6_291_456;
    const STATE_BYTES: usize = 3_145_728;
    const LMPOOL_BYTES: usize = 542_113_792;
    const ACT_BYTES: usize = 1_048_576;

    /// Everything device-side, held across requests. Raw XRT handles are not
    /// thread-affine; this struct is only ever used by one thread at a time
    /// (the server serializes all generation behind its backend `Mutex`), it
    /// just needs to MOVE into the server thread — hence the manual `Send`.
    struct Resident {
        // field order = drop order: runs/bos before kernels/contexts/device.
        pools: Vec<Bo>,
        packs: Vec<Bo>,
        sides: Vec<Bo>,
        states: Vec<Bo>,
        lmpool: Bo,
        act: Bo,
        logits: Bo,
        k_l: Kernel,
        k_l2: Kernel,
        k_lm: Kernel,
        _ctx_l: Context,
        _ctx_l2: Context,
        _ctx_lm: Context,
        _dev: Device,
        _lock: NpuLock,
    }

    unsafe impl Send for Resident {}

    fn load_bo(dev: &Device, size: usize, init_file: Option<&std::path::Path>) -> Result<Bo, String> {
        let mut bo = dev.bo(size)?;
        match init_file {
            Some(p) => {
                let data = std::fs::read(p).map_err(|e| format!("read {}: {e}", p.display()))?;
                bo.init(&data)?;
            }
            None => bo.init(&[])?,
        }
        bo.sync_to_device()?;
        Ok(bo)
    }

    impl Resident {
        fn open(cfg: &L40Config) -> Result<Resident, String> {
            let lock = NpuLock::acquire("l40")?;
            let t0 = Instant::now();
            let dev = Device::open(0)?;
            eprintln!("l40: device {} open", dev.name());
            // Two contexts on the SAME layer.xclbin: register_xclbin is
            // idempotent, so the second hwctx() reuses the registration and
            // just creates the ping-pong partner context.
            let ctx_l = dev.hwctx(&cfg.xclbin_dir.join("layer.xclbin"))?;
            let ctx_l2 = dev.hwctx(&cfg.xclbin_dir.join("layer.xclbin"))?;
            let ctx_lm = dev.hwctx(&cfg.xclbin_dir.join("lm_head.xclbin"))?;
            let k_l = ctx_l.kernel(&cfg.elf_dir.join("elf_000005.bin"))?;
            let k_l2 = ctx_l2.kernel(&cfg.elf_dir.join("elf_000005.bin"))?;
            let k_lm = ctx_lm.kernel(&cfg.elf_dir.join("elf_000003.bin"))?;

            let n = cfg.num_layers;
            let mut pools = Vec::with_capacity(n);
            let mut packs = Vec::with_capacity(n);
            let mut sides = Vec::with_capacity(n);
            let mut states = Vec::with_capacity(n);
            for l in 0..n {
                pools.push(load_bo(&dev, POOL_BYTES, Some(&cfg.buf_dir.join(format!("pool_L{l}.bin"))))?);
                packs.push(load_bo(&dev, PACK_BYTES, Some(&cfg.buf_dir.join(format!("pack_L{l}.bin"))))?);
                sides.push(load_bo(&dev, SIDE_BYTES, Some(&cfg.buf_dir.join(format!("side_L{l}.bin"))))?);
                states.push(load_bo(&dev, STATE_BYTES, None)?);
                if l % 10 == 9 {
                    eprintln!("l40: {}/{n} layer pools resident ({:.0}s)", l + 1, t0.elapsed().as_secs_f64());
                }
            }
            let lmpool = load_bo(&dev, LMPOOL_BYTES, Some(&cfg.buf_dir.join("pool_lmhead.bin")))?;
            let act = load_bo(&dev, ACT_BYTES, None)?;
            let logits = load_bo(&dev, ACT_BYTES, None)?;
            eprintln!("l40: resident ({:.0}s total)", t0.elapsed().as_secs_f64());
            Ok(Resident {
                pools, packs, sides, states, lmpool, act, logits,
                k_l, k_l2, k_lm,
                _ctx_l: ctx_l, _ctx_l2: ctx_l2, _ctx_lm: ctx_lm,
                _dev: dev, _lock: lock,
            })
        }

        /// Re-zero the 40 state buffers — the whole per-request reset.
        fn zero_states(&mut self) -> Result<(), String> {
            for st in &mut self.states {
                st.init(&[])?;
                st.sync_to_device()?;
            }
            Ok(())
        }

        /// One decode step: write `act_bytes`, run the 14 pipelined ping-pong
        /// layer chunks, optionally run the NPU lm_head and return its
        /// full-vocab f32 logits (padding rows masked to -inf).
        fn step(&mut self, act_bytes: &[u8], with_logits: bool) -> Result<Option<Vec<f32>>, String> {
            self.act.write(act_bytes, 0)?;
            self.act.sync_to_device()?;

            let n = self.pools.len();
            // Keep each chunk's runs alive until that chunk's runlist is
            // waited (XRT holds references into them).
            let mut pending: Option<(Runlist, Vec<Run>)> = None;
            for (g, c0) in (0..n).step_by(3).enumerate() {
                let (ctx, k) = if g % 2 == 0 { (&self._ctx_l, &self.k_l) } else { (&self._ctx_l2, &self.k_l2) };
                let mut rl = ctx.runlist()?;
                let mut runs: Vec<Run> = Vec::with_capacity(3);
                for l in c0..(c0 + 3).min(n) {
                    let mut r = k.run()?;
                    r.set_arg_int(0, 3)?;
                    r.set_arg_int(1, 0)?;
                    r.set_arg_int(2, 0)?;
                    r.set_arg_bo(3, &self.pools[l])?;
                    r.set_arg_bo(4, &self.act)?;
                    r.set_arg_bo(5, &self.packs[l])?;
                    r.set_arg_bo(6, &self.sides[l])?;
                    r.set_arg_bo(7, &self.states[l])?;
                    rl.add(&r)?;
                    runs.push(r);
                }
                rl.execute()?; // submit; do NOT wait yet (M5 servep pipelining)
                if let Some((mut prl, _pruns)) = pending.take() {
                    prl.wait()?;
                }
                pending = Some((rl, runs));
            }
            if let Some((mut prl, _pruns)) = pending.take() {
                prl.wait()?;
            }

            if !with_logits {
                return Ok(None);
            }
            let mut r = self.k_lm.run()?;
            r.set_arg_int(0, 3)?;
            r.set_arg_int(1, 0)?;
            r.set_arg_int(2, 0)?;
            r.set_arg_bo(3, &self.logits)?;
            r.set_arg_bo(4, &self.lmpool)?;
            r.set_arg_bo(5, &self.act)?;
            r.start()?;
            let st = r.wait()?;
            if st != STATE_COMPLETED {
                return Err(format!("lm_head run ended in state {st}"));
            }
            self.logits.sync_from_device()?;
            let mut raw = vec![0u8; LOGITS_BYTES];
            self.logits.read(&mut raw, 0)?;
            // Truncate to the real vocab: rows [REAL_VOCAB..VOCAB) are
            // lm_head padding with undefined content, and the sampler
            // (correctly) refuses non-finite logits.
            let lg: Vec<f32> = raw[..REAL_VOCAB * 2]
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            Ok(Some(lg))
        }
    }

    /// Resident 40-layer NPU backend: NPU prefill (decode-as-prefill) + NPU
    /// decode + NPU lm_head. Holds the `.q4nx` mmap (for embedding rows and
    /// the final-norm weight only) and the resident device state across calls.
    pub struct L40Backend {
        cfg: L40Config,
        file: Option<Q4nx>,
        norm_weight: Vec<f32>,
        resident: Option<Resident>,
    }

    impl L40Backend {
        pub fn new(cfg: L40Config) -> L40Backend {
            L40Backend { cfg, file: None, norm_weight: Vec::new(), resident: None }
        }

        pub fn open_default() -> L40Backend {
            L40Backend::new(L40Config::default())
        }

        fn ensure_open(&mut self) -> Result<(), String> {
            if self.file.is_none() {
                let f = Q4nx::open(&self.cfg.model_path)
                    .map_err(|e| format!("open {}: {e}", self.cfg.model_path.display()))?;
                self.norm_weight = f.bf16("model.norm.weight");
                self.file = Some(f);
            }
            if self.resident.is_none() {
                self.resident = Some(Resident::open(&self.cfg)?);
            }
            Ok(())
        }

        fn act_for(&self, token: u32) -> Vec<u8> {
            let file = self.file.as_ref().expect("model open");
            let embed: Vec<f64> = file
                .embed_row("model.embed_tokens.weight", token as usize, HIDDEN)
                .iter()
                .map(|v| *v as f64)
                .collect();
            state_io::write_act(&embed, &self.norm_weight)
        }
    }

    impl Backend for L40Backend {
        fn generate(
            &mut self,
            prompt_ids: &[u32],
            params: &mut GenParams,
            on_token: &mut dyn FnMut(u32),
        ) -> Result<(), String> {
            if params.max_tokens == 0 {
                return Ok(());
            }
            if prompt_ids.is_empty() {
                return Err("empty prompt".to_string());
            }
            if prompt_ids.len() + params.max_tokens > MAX_POSITIONS {
                return Err(format!(
                    "prompt ({}) + max_tokens ({}) exceeds the {MAX_POSITIONS}-position KV capacity",
                    prompt_ids.len(),
                    params.max_tokens
                ));
            }
            self.ensure_open()?;
            let stop_ids = params.stop_ids.clone();
            let is_stop = |id: u32| stop_ids.contains(&id) || tokenizer::EOS_TOKEN_IDS.contains(&id);

            // A step error leaves device state unknown; drop residency so the
            // next request reopens from scratch instead of continuing on it.
            let result = (|| {
                let mut history: Vec<i64> = prompt_ids.iter().map(|&t| t as i64).collect();

                // ---- NPU prefill: sequential decode from zeroed states ----
                let t0 = Instant::now();
                self.resident.as_mut().unwrap().zero_states()?;
                let mut logits: Option<Vec<f32>> = None;
                for (i, &tok) in prompt_ids.iter().enumerate() {
                    let act = self.act_for(tok);
                    let last = i == prompt_ids.len() - 1;
                    logits = self.resident.as_mut().unwrap().step(&act, last)?;
                }
                let lg0 = logits.expect("final prefill step returns logits");
                eprintln!(
                    "l40: prefill {} tokens in {:.2}s ({:.0} ms/tok)",
                    prompt_ids.len(),
                    t0.elapsed().as_secs_f64(),
                    t0.elapsed().as_secs_f64() * 1000.0 / prompt_ids.len() as f64
                );
                if !lg0.iter().all(|v| v.is_finite()) {
                    return Err("non-finite prefill logits".to_string());
                }

                // ---- sample + NPU decode loop -----------------------------
                let t1 = Instant::now();
                let mut emitted = 0usize;
                let first = params.sampler.sample(&lg0, &history) as u32;
                if is_stop(first) {
                    return Ok(());
                }
                on_token(first);
                history.push(first as i64);
                let mut cur = first;
                emitted += 1;

                while emitted < params.max_tokens {
                    let act = self.act_for(cur);
                    let lg = self
                        .resident
                        .as_mut()
                        .unwrap()
                        .step(&act, true)?
                        .expect("step(with_logits) returns logits");
                    let nxt = params.sampler.sample(&lg, &history) as u32;
                    if is_stop(nxt) {
                        break;
                    }
                    on_token(nxt);
                    history.push(nxt as i64);
                    cur = nxt;
                    emitted += 1;
                }
                if emitted > 1 {
                    eprintln!(
                        "l40: {} decode tokens at {:.2} tok/s",
                        emitted,
                        (emitted - 1) as f64 / t1.elapsed().as_secs_f64()
                    );
                }
                Ok(())
            })();
            if result.is_err() {
                self.resident = None;
            }
            result
        }
    }
}

#[cfg(feature = "npu")]
pub use npu_impl::L40Backend;

#[cfg(all(test, feature = "npu"))]
mod tests {
    use super::*;
    use crate::backend::{Backend, GenParams};
    use crate::sampler::Sampler;
    use crate::tokenizer::{ChatMessage, Tokenizer, EOS_TOKEN_IDS};
    use std::collections::HashSet;

    /// End-to-end, real-hardware smoke test for the FULL Phase-1 path:
    /// tokenizer + chat template -> NPU decode-as-prefill -> NPU decode with
    /// NPU lm_head -> detokenize. Needs the base model, prebuilt m3out/l40
    /// buffers and the real NPU (~21 GB resident). TWO generates on one
    /// backend prove the per-request state re-zero works. Run with:
    ///   cargo test --release --features npu --bin open-qwen-npu \
    ///     generate_l40:: -- --ignored --nocapture
    #[test]
    #[ignore]
    fn generates_twice_on_real_hardware() {
        let cfg = L40Config::default();
        let model_dir = cfg.model_path.parent().expect("model dir").to_path_buf();
        let tok = Tokenizer::load(&model_dir).expect("load tokenizer");
        let mut backend = L40Backend::new(cfg);

        for prompt_text in ["Say hi.", "What is the capital of France?"] {
            let msgs = vec![ChatMessage { role: "user".to_string(), content: prompt_text.to_string() }];
            let prompt = tok.apply_chat_template(&msgs, true, true).expect("render chat template");
            let ids = tok.encode(&prompt).expect("encode prompt");
            println!("prompt ({} tokens): {prompt_text:?}", ids.len());

            let sampler = Sampler::new(0.0, 0, 1.0, 1.0, None); // greedy
            let stop_ids: HashSet<u32> = EOS_TOKEN_IDS.iter().copied().collect();
            let mut params = GenParams { max_tokens: 12, sampler, stop_ids };

            let mut out_ids: Vec<u32> = Vec::new();
            backend.generate(&ids, &mut params, &mut |t| out_ids.push(t)).expect("generate");
            assert!(!out_ids.is_empty(), "no tokens generated for {prompt_text:?}");
            println!("generated: {:?}", tok.decode(&out_ids));
        }
    }
}
