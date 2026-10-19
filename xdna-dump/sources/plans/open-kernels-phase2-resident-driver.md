# Open kernels phase 2, item 4: the resident driver (`OpenBackend`)

**Status: plan (2026-09-03).** Follows item 3, dynamic KV length
(`open-kernels-phase2-dynamic-kv.md`, commit 1ce32fa). Goal from the
feasibility plan: Josh's pruned 27B (30L, interval 3) served end to end on the
open kernels through the same `Backend` / HTTP / sampler path `L40Backend`
uses today, with nothing closed on the request path.

## Spec impact

The repo has no `specs/` tree yet; this is the first feature that gets one:
`specs/open-engine/spec.md`, prefix `OPEN`. Initial requirements (all
hardware-bound, so **Test category: integration** with `#[ignore]` cargo
tests in the crate, listed under **External tests**; the acceptance criteria
are the verification artifact):

- **OPEN-GENERATE**: `OpenBackend::generate` produces tokens for a prompt with
  no closed kernel and no CPU forward math on the request path (embedding
  rows and the final-norm weight are the only model reads on the host).
  Acceptance: greedy generation from the single-token prompt `[248045]`
  yields `846, 198, 3710` (the replica's greedy sequence, item 3).
- **OPEN-PREFILL**: prompt tokens are prefilled by sequential decode from
  zeroed states (decode-as-prefill; exact for these kernels because the
  position is host-driven per token). Acceptance: the first sampled token
  after a T-token prompt equals the token the item-3 harness gives for the
  same T positions.
- **OPEN-RESET**: a second `generate` on the same backend starts from zeroed
  linear states and caches. Acceptance: two identical requests give
  identical token sequences.
- **OPEN-CAPACITY**: `prompt + max_tokens > MAX_CTX (4096)` is rejected
  before touching the device.
- **OPEN-LOGITS**: logits above the real vocab (248070) are masked before
  sampling; non-finite logits are a hard error.

## Design

New module `npu-engine/src/generate_open.rs`, modelled on `generate_l40.rs`
(same `Resident` shape, same `Backend` impl skeleton, same error policy: a
step error drops residency).

**Config** `OpenConfig { model_path, buf_dir, kernel_dir, num_layers,
layer_types }`: `buf_dir` = the existing l30 build dir
(`m3out/l30_27b`: `pool_L{l}.bin`, `pack_L{l}.bin`, `side_L{l}.bin`,
`pool_lmhead.bin` — already prompt-independent, already the pools.rs layout
the open kernels read), `kernel_dir` = `tools/open-kernels/designs`
(`layer_x/build_lx0|lx1|ax0|ax1`, `ln/build`, `lm_head_q8/build_full`: the
xclbin + `insts.bin` pairs). Env overrides `OPEN_QWEN_OPEN_BUF_DIR`,
`OPEN_QWEN_KERNEL_DIR`, model via `OPEN_QWEN_L30_MODEL` as today.

**Resident**: device + lock; contexts X (`build_lx0/final.xclbin`), Y
(`build_ax0`), L (`ln`), K (`lm_head_q8`); kernels via
`Context::kernel_xclbin("MLIR_AIE")`; six instruction BOs (`bo_instr`) —
lx0, lx1, ax0, ax1, ln, lm. Per layer: pool (512 MB), consts (linear 11.9 MB
/ attention 1.06 MB), state (2.3 MB, linear) or kv (8 MB, attention), act
(190 / 98 KB). Shared: xres, zero, normw, xresf, hn (8/8/4/8/4 KB), logits
(993 280 B f32), ptab (4 MB), lmpool (542 MB). ~16.2 GB resident.

**Consts builder** (`pools.rs`, `build_open_consts(pack, side, full)`): the
exact slicing `make_27b.py` does today (linear: `[lnw | glue side minus xn |
nw | postln | router W | sgw | out_proj]`; attention: `[lnw | postln | qn kn
| router W | sgw]`), from the pack/side *bytes* so the Python and Rust
builders can be byte-compared. `ptab` built in Rust (same bytes as
`layout.ptab()`).

**Stream patching** (`npu-engine/src/patch.rs`, refactored out of
`decode.rs` with no behaviour change to `npu` configs): `moe2_table(words)`,
`moe2_apply(ibo, table, idx)`, `attn_table(words)`, `attn_apply(ibo, table,
pos)` — pure functions over the instruction words; `Driver` calls them.

**Step** (token `tok` at position `p`, `with_logits`):
1. `xres.write(embed(tok) as f32[2048])`, sync.
2. if `p` changed since the last step: `attn_apply(ax0, p)` (one sync).
3. per layer `l`: run part 0 (`lx0 pool xres consts state act` /
   `ax0 pool xres consts kv act ptab`), `act.sync_from_device`, read the
   int32 idx[8] at `A_ROUT + 1024` / `AA_ROUT + 1024`, `moe2_apply(part1,
   idx)` (sync), run part 1. Submit + wait per run (`start`/`wait`, state 4
   or error), as the driver's `run` does.
4. `ln xres zero normw -> xresf hn`; if `with_logits`: `lm lmpool hn ->
   logits`, sync, read 248070 f32.

62 submits per token, ~155 ms (item 3's numbers): ~6.4 tok/s decode and the
same per prompt token (a 40-token prompt is ~6 s). Runlist batching is not
possible across a part boundary (the routing read is a host step); folding
the MoE part via on-device expert fetch (0b) is the later item that removes
it.

**Prefill**: decode-as-prefill, positions `0..T-1`, logits only on the last.
Per request: zero the 20 state BOs and the 10 kv BOs (`init(&[])` +
sync, ~130 MB).

**CLI / server**: `open-qwen-npu open-run [prompt] [max_tokens]` (through
`run_backend_cli`), `serve --backend open` (model id `qwen3.6-27b-a2.8b`).
Process exit: XRT tears down 4 contexts with a segfault after all work
(known, harmless); `open-run` flushes stdout and calls `process::exit(0)`
after generation so the CLI returns 0.

## Files

- `npu-engine/src/generate_open.rs` (new): `OpenConfig`, `Resident`,
  `OpenBackend`, `#[ignore]` hardware tests (OPEN-GENERATE/PREFILL/RESET).
- `npu-engine/src/patch.rs` (new): the two patch tables + appliers;
  `decode.rs` switches to them.
- `npu-engine/src/pools.rs`: `build_open_consts`, `build_ptab`; unit tests
  byte-compare against `designs/decode_chain/w27/consts2_0.bin`,
  `constsa2_2.bin`, `ptab.bin` when those files exist (skip otherwise).
- `npu-engine/src/main.rs`: `open-run`, `serve --backend open`, usage.
- `specs/open-engine/spec.md` (new) + `specs/open-engine/tests/README.md`
  pointing at the crate's tests (Rust: the tests live in the crate; the spec
  lists them as External tests).
- `tools/open-kernels/README.md`: the "Running through phlegm" section gains
  the backend.

## Verification

1. `cargo test --features npu` (no hardware): consts / ptab byte equality
   with the Python builders' files; patch tables on the built `insts.bin`
   (window/drain/record words found; moe2 placeholder count).
2. Hardware: `cargo test --release --features npu generate_open:: --
   --ignored --nocapture`: `[248045]` greedy → `846, 198, 3710`; two
   generates on one backend identical; a chat-template prompt decodes to
   coherent text at ~6 tok/s (reported).
3. `serve --backend open` + one `curl /v1/chat/completions`.
4. Position 0 logits of the backend's first step vs `w27/ref_logits.bin`
   (corr 0.999998 expected) — a debug env `OPEN_QWEN_DUMP_LOGITS=<path>`
   dumps the first step's logits for `compare_27b.py`.

## Out of scope

Prefill batching (many rows per step), folding the MoE part (0b expert
fetch), runlist pipelining, the even/odd lm_head split (the open lm_head is
full-vocab f32), tokenizer/server changes.
