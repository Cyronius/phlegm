# Open kernels phase 3: model recipes (compose the kernels per family, don't rewrite them)

**Status: plan (2026-09-03).** Sits alongside phase 2 item 4, the resident
driver (`open-kernels-phase2-resident-driver.md`), and changes what item 4
should read its configuration from. Nothing here is implemented.

## The problem, stated precisely

llama.cpp supports a new model family with one commit: a `build_<arch>()`
graph function composing ~30 shape-polymorphic ops. New *ops* (Mamba's scan,
Qwen3-Next's gated delta) still need kernel commits there too. That is the
floor for any engine; the goal is to get ours down to it.

We are above that floor for one reason, and it is not "the NPU is weaker":
**on XDNA2 composition happens at compile time.** Shapes, tile placement, DMA
programs and fusion are all baked into the xclbin, and phase 2 measured why
you cannot recover runtime composition by dispatching one small xclbin per op:
the whole-layer context took the 27B step from 208 to 165 ms (132 → 62
dispatches), a context switch resets the cores, and the shim BD budget /
16 KB program memory are physical placement limits, not tunables. So the
"missing layer" is a **compile-time generator**: a per-family recipe that
composes reusable kernel templates into per-layer designs, built once per
(family, shape, quant) and cached — CUDA-JIT-style, at install time.

Three tiers of change, in decreasing frequency, are the target:

| change | what it costs | llama.cpp analogue |
|---|---|---|
| new model in a known family | nothing committed: new hyperparameters → rebuild + cache | new GGUF |
| new family | a recipe (Python, ~200 lines) | `build_<arch>()` |
| new op type | a kernel template | a new ggml op |

## Where the model is hard-coded today (the inventory to lift)

Everything below is Qwen3.6-27B-specific by construction. Listed by layer of
the stack, innermost first, with what a parameter would look like.

1. **Kernel bodies** (`tools/open-kernels/designs/*/*.h|.cc`). These are the
   template pieces and are already mostly parametric:
   - `gemv_q4/gemv_q4.h`: K is a runtime argument; `GEMV_PER_CALL`,
     `GEMV_PER_BAND`, `GEMV_ROWSPLIT` compile-time. Bakes: **q4_1 nibbles in
     FLM's pool-chunk order** (`pools.rs std_perm`, 64-row bands of K/128
     chunks) and the int16 activation table for K ∈ {2048, 4096}
     (`gemv_q4_prep_k{K}`). Missing knobs: Q4_K (`qwen36-1.0.3-format-support.md`),
     other K, q8 (lm_head has its own design).
   - `attn/attn.h`: HD = 256, rotary 64 of 256 half-split, θ = 1e7, qk RMSNorm
     with weights, sigmoid gate, KV row = [K bf16 512 | V bf16 512]. Missing
     knobs: HD ∈ {64, 128, 256}, full vs partial RoPE + θ + Llama-3 frequency
     scaling, no gate, no qk-norm, sliding window (a row-range, see phase C),
     GQA ratio (already implicit through the projection widths).
   - `layer_x/dnx*.cc` (DeltaNet), `dn_glue`, `dn_post`: 32 heads × 128. Only
     Qwen3-Next/3.5/3.6 families use these; they stay as-is.
   - `ln`: RMSNorm + fused residual, width 2048. Knob: width; Gemma's
     `(1 + w)` weight convention is a host-side transform, not a kernel change.
   - `router`, `moe_*`: 256 experts, top-8, softmax + renorm, 8 routed slots,
     one shared expert with sigmoid gate. Knobs: E, top-k, shared-expert
     presence, sigmoid vs softmax routing (DeepSeek/Qwen3 differ).
   - `silu_mul`: knob: activation (SiLU / GeGLU-tanh for Gemma).
   - `lm_head_q8`: 1940 bands hand-split 243/242 over 8 cores. Knob: vocab.

2. **Design geometry** (`layer_x/lx.py`, `layer_x/ax.py`): `HID = 2048`,
   `NH, KVH, HD = 16, 2, 256`, `N_CORES = 8`, bands per core derived
   (`QKV_PC = 8192 // 64 // 8`, …), helper cores hand-placed at `Tile(c, 2)`,
   `Tile(0, 3)`, `Tile(1, 3)`, `Tile(2, 3)`, shim budget 13 fills / 11 drains
   (`open-kernels-phase2-whole-layer.md:35`), 16 KB program memory per core,
   ≤ 8 buffer arguments per run (ERT state 6 above that). The first three are
   arithmetic on the spec; the last four are **constraints a generator must
   check, not derive** — the honest limit of "parametrize".

3. **Byte layouts** (`layer_x/layout.py`): consts/act/state offsets, pool
   offsets (`POOL_QKV = 505_282_560`, …), `KV_ROW = 2048`, `PTAB_ROW`,
   `MAX_CTX = 4096`, the RoPE table. **Mirrored by hand in Rust**:
   `npu-engine/src/decode.rs:56-70` (`MOE_*`, `ATTN_*`) and
   `pools.rs` (`CH = 5120`, `std_perm`). Two copies of one truth; the
   generator must emit the one the driver reads.

4. **Pool packing** (`make_27b.py` / `build_pools.py` / `pools.rs`): which
   tensor lands at which pool offset in which chunk order — FLM's layer-pool
   layout, chosen so the closed kernels could share files. New families owe
   FLM nothing; the packer can be recipe-driven.

5. **The step program** (`layer_x/run.cfg`, and item 4's `Step`): per layer
   `run lx0` → `moeroute2 lx1 <off>` → `run lx1` (linear) or `attnpos ax0
   <pos>` → `run ax0` → `moeroute2` → `run ax1` (attention), then `ln`, `lm`.
   The driver already has the verb set (`decode.rs`: `run`, `runlist`,
   `moeroute2`, `attnpos`, `copy`, `setwords`, `ctrlpkt`, …). What is
   hard-coded is *which sequence*, in `generate_open.rs` (item 4) and
   `OpenConfig { num_layers, layer_types }`.

6. **The model file and its metadata**: `q4nx` tensor names are Qwen's;
   `tools/q4nx-convert/convert.py:305-315` writes a `config.json` with
   `qwen35moe.*` keys only; `L40Config` defaults to literal paths.

## Design

### Artifacts

**`ModelSpec`** — the hyperparameter tuple, derived from GGUF metadata (the
one place every family already publishes it: `general.architecture`,
`<arch>.embedding_length`, `.attention.head_count(_kv)`, `.rope.*`,
`.attention.sliding_window`, `.expert_count`, …). Plain dataclass, JSON on
disk, per-layer `layer_types` explicit (Qwen3.6's interval pattern, Gemma's
5:1 local/global, dense = all one type).

**Op catalogue** — each kernel template declares the parameters it takes and
the ranges it has been *validated* for (`K ∈ {2048, 4096}` today, not "any
K"). A recipe asking for an unvalidated point fails at generation time with
the constraint named, never at run time with garbage. This is where the
program-memory / BD-budget / ≤ 8-args facts live as checks.

**Family recipe** — `tools/open-kernels/recipes/<family>.py`:
`ModelSpec → [LayerDesign]`, one `LayerDesign` per distinct layer type. It
composes catalogue ops into the streams lx/ax use today (w / x / y per main
core, helpers by role), computes the byte layout, and emits:
- the IRON design source (today's `lx.py` / `ax.py` become the
  `qwen36moe` recipe's two outputs, parametrized);
- `layout.json`: every offset the driver needs (replaces the Rust mirror);
- the pool packing plan (tensor → pool offset → chunk order);
- the **step program**: per layer type, the verb sequence with patch points
  (`moeroute2` offset, `attnpos` stream, buffer arg lists).

**Manifest** — `manifest.json` next to the built xclbins: spec hash,
contexts and their kernels, layout, step program, BO sizes. **The driver
reads only this.** No `HID`, no `POOL_*`, no `lx0`/`ax0` names in Rust.

**Build cache** — key = hash(recipe source, kernel sources it includes,
ModelSpec, quant format, `MAX_CTX`). `phlegm build-kernels <model>` runs the
generator + `build_design.py` per LayerDesign in WSL (minutes per design;
the 27B has two, dense families one) into `~/.phlegm/kernels/<hash>/`.
`OpenBackend` refuses to start on a missing or mismatched manifest — no
silent fallback to a closed path.

### The driver becomes an interpreter

Item 4's `Step` today is: embed → per layer (hard-coded verb pairs) → ln →
lm. Under this plan it is: embed → for each layer, execute the manifest's
program for `layer_types[l]` with the layer's BOs bound by name → tail
program. The verbs are the ones `decode.rs` already runs from `run.cfg`;
`patch.rs` (item 4) gains nothing new. **Recommendation: build item 4
against a hand-written `manifest.json` for the 27B from day one**, generated
by a 40-line script from today's `layout.py` constants. Item 4 is then not
blocked on the Python parametrization (phase A), and phase A's gate is
"the generated manifest is byte-identical to the hand-written one".

### On-disk format for new families

The q4nx / FLM pool order exists so the closed kernels could read our files.
New families never touch the closed path, so: **the recipe's packer writes
`pool_L{l}.bin` + `consts_L{l}.bin` straight from GGUF** in whatever order
its GEMV template wants (today's `std_perm` for q4_1; the Q4_K order when
that template lands). q4nx stays for the closed backends only. This also
removes the converter's per-family name-map burden — the packer reads GGUF
tensor names via the recipe.

## Phasing

**Phase A — parametrize what exists (no new model).** Extract `ModelSpec`
for the 27B; `lx.py` / `ax.py` / `layout.py` read it; `gen_kernels.py`
emits per-spec kernel TUs; emit `manifest.json`; `decode.rs` / `pools.rs`
lose their mirrored constants. Gate: rebuilt xclbins + `insts.bin`
byte-identical to today's (or, if IRON's cache keying changes the bytes,
27B logits corr ≥ 0.999998 vs the replica and ≤ 165 ms/step). Also lifts
`attn.h` to `HD ∈ {128, 256}` and `gemv_q4_prep` to `K ∈ {2048, 2560, 3072,
4096}`, validated with the existing `make_test.py` / `compare.py` per design
at each new point — this is the work, and it is unglamorous.

**Phase B — first dense family: Qwen3 dense (4B: HID 2560, 32/8 heads,
HD 128; or 8B: 4096, 32/8, HD 128).** Closest to what exists: same GEMV
order, same tokenizer family, qk-norm already in `attn.h`, no MoE, no
DeltaNet. New design `dx` (dense layer) = ln → gemv q|k|v → attn → gemv o →
ln → gemv up|gate → silu_mul → gemv down, one instruction stream (no routing
read, so no part split: fewer dispatches than the MoE layers). Oracle:
llama.cpp logits for the same GGUF (`compare_logits` tool, generic; the CPU
replica is Qwen3.6-only and stays that way). Gate: greedy match over 32
tokens from a chat-template prompt.

**Phase C — Llama 3 8B: the "recipe-only" proof.** No qk-norm, no gate,
full RoPE θ = 5e5 with Llama-3 frequency scaling (a table in `ptab`, host
side), vocab 128256. Should be a recipe commit and zero kernel commits; if it
isn't, the catalogue's knobs are wrong and phase B's parametrization is
fixed before continuing.

**Phase D — Gemma 3 (4B / 12B): the "new op" proof**, and the model FLM
users are hitting. New ops: GeGLU-tanh (`silu_mul` gains an activation knob),
sliding-window attention (the KV fill becomes rows `[max(0, pos − W), pos)`
— `attnpos` already patches length and offsets, so this is a driver
formula plus a mask bound in `attn.h`), sandwich norms (post-attn and
post-FFN — two extra `ln` runs, no new kernel), embedding scale √HID and
`(1 + w)` norm weights (host-side transforms), 262 144-vocab tied lm_head
(1.1 GB q8 pool; memory, not code), HD 256 (exists), θ = 1e6 global / 1e4
local per layer type. Five of those are recipe-level; two touch kernels.

Prefill batching (many rows per step) is orthogonal and stays in phase 2.

## Spec impact

New feature spec `specs/open-engine/spec.md` (prefix `OPEN`, the one item 4
creates) gains:

- **OPEN-MANIFEST** — `OpenBackend` derives every layout constant, context,
  and per-layer verb sequence from `manifest.json`; a missing or
  spec-mismatched manifest is a startup error. *unit* (manifest parse +
  mismatch rejection with a fixture) + *integration* (27B greedy `[248045]` →
  `846, 198, 3710` through the manifest path — item 4's OPEN-GENERATE
  becomes this).
- **OPEN-SPEC-DERIVE** — `ModelSpec` from GGUF metadata for each supported
  arch; unknown arch is an error naming the key. *unit* (fixtures: the
  metadata dicts of one GGUF per family).
- **OPEN-BUILD-CACHE** — cache key covers recipe source, included kernel
  sources, spec, quant, `MAX_CTX`; any change rebuilds. *unit*.
- **OPEN-OP-RANGE** — a recipe requesting an op outside its validated
  parameter set fails at generation with the constraint named. *unit*.
- **OPEN-FAMILY-QWEN36MOE / -QWEN3 / -LLAMA3 / -GEMMA3** — one per family:
  greedy agreement with the oracle over N tokens on the named reference
  model. *integration*, `#[ignore]` hardware tests, listed as External tests
  in the spec.

## Files

- `tools/open-kernels/recipes/{spec.py, catalogue.py, qwen36moe.py,
  qwen3.py, llama3.py, gemma3.py}` (new; the last three per phase).
- `tools/open-kernels/designs/layer_x/{lx,ax,layout}.py` → parametrized by
  `ModelSpec`; `designs/dx/` (new, phase B).
- `tools/open-kernels/gen_manifest.py` (new; phase A stopgap: hand-written
  from today's constants, later emitted by the recipe).
- `tools/open-kernels/pack_pools.py` (new): GGUF → `pool_L{l}.bin` per the
  recipe's plan; `make_27b.py` becomes a caller.
- `npu-engine/src/manifest.rs` (new), `generate_open.rs` (item 4) reads it;
  `decode.rs:56-70` and `pools.rs` constants removed.
- `npu-engine/src/main.rs`: `build-kernels <gguf>` (invokes WSL), `open-run`
  takes a model dir with a manifest.
- `tools/compare_logits.py` (new): our logits vs llama.cpp's for one GGUF.
- `specs/open-engine/spec.md` + tests.

## Verification

1. Phase A: byte-diff of generated vs hand-written manifest; xclbin/insts
   byte-diff; 27B greedy sequence; step time unchanged.
2. Per new op parameter point: the design's `make_test.py` + `compare.py`
   (cos, maxrel) before any layer design uses it.
3. Per family: `compare_logits.py` at position 0 (corr ≥ 0.999) then greedy
   over 32 tokens vs llama.cpp on the same GGUF at the same quant.
4. `cargo test --features npu` (no hardware): manifest, spec-derive, cache
   key, op-range.

## Decisions for Josh

1. **Phase B family**: Qwen3 dense (recommended, closest) vs Llama 3 first.
2. **Item 4 ordering**: land item 4 against a hand-written manifest now
   (recommended), or wait for phase A.
3. **Drop q4nx for new families** (recipe packs pools from GGUF directly;
   recommended) vs extend the converter per family.
4. **Build-at-install in WSL** is acceptable for now; a native Windows
   IRON build is its own project and not in this plan.

## Out of scope

Auto-placement beyond per-family templates (mlir-aie's job, immature),
prefill batching, Q4_K kernels (own plan), native Windows toolchain, vision
towers, anything closed-kernel.
