# GGUF → q4nx converter (Qwen3.6-MoE, open NPU engine)

Converts a **Qwen3.6-MoE GGUF** (llama.cpp `qwen35moe` arch, e.g. Cyrus's
[Cyronius/Qwen3.6-27B-A2.8B](https://huggingface.co/Cyronius/Qwen3.6-27B-A2.8B)
`Q4_K_M`) into the **q4nx FILE format** that the open NPU engine (`npu-engine/`)
and the reference forward (`tools/kernel-interp/full_forward.py`) read.

It exists because the official converter
([FastFlowLM/FLM_Q4NX_Converter](https://github.com/FastFlowLM/FLM_Q4NX_Converter))
now targets a **newer** FLM format (Q4_K super-blocks, 4736-byte chunks, plain
column-major within-chunk nibbles). Cyrus's installed engine (flm 1.0.2) — the one
the open engine was reverse-engineered from — reads the **older q4_1 format**
(5120-byte chunks, bf16 scale+min, a 16-lane nibble interleave) plus a q8 lm_head.
This converter emits exactly that verified layout.

```
python convert.py -i qwen36-27b-a2.8b-mtp-Q4KM.gguf -o out_dir
#   -> out_dir/model.q4nx   (+ config.json)
python convert.py -i model.gguf -o out_dir --layers 3   # first-N-layers debug slice
```

## How it works

Per tensor: **GGUF k-quant block → `gguf.dequantize` → f32 → arch transform →
re-quantize (q4_1 / q8) into the FLM 16-lane layout → safetensors** (I8 / BF16 / F32).
The numeric quantization is delegated to ggml's own `quantize()` (per-32 block,
identical block scale/min choice to llama.cpp); `q4nx_format.py` only performs the
FLM byte re-layout, and stores the block scale/min as **bf16** (FLM's format) — the
only lossy step beyond the requant, ≈0.001 per weight.

### q4nx FILE byte layout (verified against Cyrus's model_3LiF.q4nx + HF weights)

A quantized `[out, in]` weight → I8 safetensors tensor `[out//32, in//256, CHUNK]`,
one CHUNK per 32-row × 256-col tile, tiles in **plain raster order**
(tile `f` → rows `32*(f//ncol)`, cols `256*(f%ncol)`, `ncol = in//256`). The engine's
loader applies the exotic pool tilings at model-load time; the FILE is raster.

- **q4_1 CHUNK = 5120 B**: `256 bf16 d` + `256 bf16 m` (planar, slot `j = bc*32+r`)
  + `4096 B` nibbles, 16-lane interleave
  `nibble[(r//16)*4096 + bc*512 + i*16 + (r%16)] = quant(row r, col bc*32+i)`.
  Per-32 block along in-dim: `value = q*d + m`.
- **q8 CHUNK = 8704 B** (lm_head): `256 bf16 d` + `8192 int8` (same 16-lane interleave,
  one byte per value); `value = q*d` (symmetric, no min).

`r`=row in tile 0..31, `bc`=32-col block 0..7, `i`=col in block 0..31.

## Verified tensor mapping (GGUF/HF → q4nx)

Every transform below was checked byte-for-byte: FLM-dequant of Cyrus's
`model_3LiF.q4nx` vs the candidate transform of the HF-original weight
(`tools/kernel-interp/hf_ref/`). Correct transform ⇒ **quant bound** (~0.01–0.03);
wrong ⇒ ~0.3–0.6.

### Quantized (I8, q4_1 5120B) — experts stored **contiguous per expert**
| GGUF name | q4nx name | shape | transform |
|---|---|---|---|
| `blk.N.attn_qkv.weight` | `…linear_attn.qkv_proj.weight` | [8192,2048] | identity |
| `blk.N.attn_gate.weight` | `…self_attn.gate_proj.weight` (z-gate) | [4096,2048] | identity |
| `blk.N.ssm_out.weight` | `…linear_attn.ssm_out_proj.weight` | [2048,4096] | identity |
| `blk.N.attn_q.weight` (full-attn) | `…self_attn.q_proj.weight` | [8192,2048] | **deinterleave** `(g16 p2 h256)→(p g h)` → planar `[q4096 | gate4096]` |
| `blk.N.attn_k/v.weight` | `…self_attn.k/v_proj.weight` | [512,2048] | identity |
| `blk.N.attn_output.weight` | `…self_attn.o_proj.weight` | [2048,4096] | identity |
| `blk.N.ffn_{gate,up}_exps.weight` | `…mlp.{gate,up}_exps_proj.weight` | [256×512,2048] | per-expert pack, experts contiguous |
| `blk.N.ffn_down_exps.weight` | `…mlp.down_exps_proj.weight` | [256×2048,512] | per-expert pack |
| `blk.N.ffn_{gate,up,down}_shexp.weight` | `…mlp.share_{gate,up,down}_exps_proj.weight` | — | identity |

The **only** non-identity weight reorder is the full-attn `q_proj` deinterleave
(the `attn_output_gate` q/gate are stored interleaved per head in HF/GGUF, planar in FLM).

### q8 (8704B) and passthrough (BF16 / F32)
| GGUF name | q4nx name | transform / dtype |
|---|---|---|
| `output.weight` | `lm_head.weight` | q8 (8704B) |
| `token_embd.weight` | `model.embed_tokens.weight` | BF16 |
| `output_norm.weight` | `model.norm.weight` | BF16 |
| `blk.N.ffn_gate_inp.weight` | `…moe_router.weight` | **.T** → [2048,256] BF16 |
| `blk.N.ssm_alpha/beta.weight` | `…linear_attn.ssm_{alpha,beta}_proj.weight` | **.T** → [2048,32] BF16 |
| `blk.N.ssm_conv1d.weight` | `…linear_attn.ssm_conv1d.weight` | squeeze **.T** → [4,8192] BF16 |
| `blk.N.ssm_a` | `…linear_attn.ssm_a` | **−exp(A_log)** (pre-baked) F32 |
| `blk.N.ssm_dt.bias` | `…linear_attn.ssm_dt.bias` | identity F32 |
| `blk.N.ssm_norm.weight` | `…linear_attn.ssm_norm.weight` | identity BF16 (**no +1**) |
| `blk.N.attn_norm.weight` | `…input_layernorm.weight` | BF16 (see norm +1) |
| `blk.N.post_attention_norm.weight` | `…post_attention_layernorm.weight` | BF16 |
| `blk.N.attn_q/k_norm.weight` | `…self_attn.q/k_norm.weight` | BF16 (see norm +1) |
| `blk.N.ffn_gate_inp_shexp.weight` | `…shared_expert_gate.weight` | squeeze BF16 |

### Norm +1 baking
FLM stores the **effective** RMSNorm weight. The zero-centered norms
(`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `model.norm`)
are stored as **HF + 1.0**; `ssm_norm` is not zero-centered and is stored raw.
**llama.cpp bakes this +1 during HF→GGUF** for zero-centered RMSNorms (as it does
for Gemma), so a real GGUF already carries it and the converter passes norms
through. If a norm arrives zero-centered (mean ≈ 0), `convert.py` prints a warning
— that GGUF did not bake the +1 and the norms would be wrong.

### MoE expert stacking
GGUF stores `ffn_{gate,up,down}_exps.weight` as one 3-D tensor `[n_expert, out, in]`.
The q4nx FILE lays experts **contiguously**: expert `e` occupies chunks
`e*(out//32) … (e+1)*(out//32)`. `convert.py` packs each expert with `pack_q4_1`
and concatenates in expert order 0…255 (gate/up separate, not gpt-oss-interleaved —
this arch keeps them as distinct tensors). Verified: converted expert 7 matches
Cyrus's FILE expert 7 (contiguous placement, quant bound).

## Assumption that needs the real GGUF to lock down
The mapping was derived against HF-original weights. It is correct for a GGUF that
(a) preserves HF row order and (b) bakes norm +1 — both **standard llama.cpp
behavior** for NEOX-rope Qwen (no q/k permute), and consistent with the reference
converter. The reference converter additionally applies ssm/qkv/z head-pairing
reorders; those were **not** needed vs HF here (they belong to the newer FLM
format). Before trusting a full 27B run, confirm on the **base 35B GGUF** against
`model.q4nx.orig` (same per-layer structure) that each tensor decodes at the quant
bound — if any projection lands at ~0.3, that GGUF applies llama.cpp's extra
permutation and the reference converter's reorderings must be re-enabled.

## Reading FLM 1.0.3 (Q4_K) files — separate from the GGUF converter above

The open engine also **reads** q4nx files emitted by the newer FLM 1.0.3 converter
(`default_tensor_type: Q4_K`), not just the 1.0.2 (q4_1) files this converter
writes. That read path lives in the engine, not here — `tools/kernel-interp/
q4nx_v103.py` (Q4_K dequant + reorder-undo) and `q4nx.py::Q4NX` (auto-detects
4736 B Q4_K vs 5120 B q4_1 and dispatches). See
`docs/qwen36-1.0.3-format-support.md` for the byte-exact spec.

1.0.3 differs from 1.0.2 three ways: Q4_K-refit quant (4736 B chunk, per-group
uint8 scales/mins over a bf16 super-scale, min subtracted), plain column-major
intra-chunk order, and head-pairing `(q g p)->(g q p)` reorders baked into the
linear-attention tensors (undone on read). lm_head stays q8/8704 B but
column-major.

Validation tools here (need the vendored reference converter under `reference/`,
gitignored — clone from FastFlowLM/FLM_Q4NX_Converter):
- `validate_v103.py` — unit test: drives the reference's real `_pack_q4k`/
  `_pack_q8nx`/`_refit_one_side` and confirms the reader inverts it (refit floor).
  gguf 0.19 cannot ENCODE Q4_K, so this drives the packer directly (no GGUF).
- `transcode_102_to_103.py` — turns a 1.0.2 file into a genuine 1.0.3 file (for
  end-to-end forward-equivalence testing without a real 1.0.3 sample).
- `compare_readback.py` — per-tensor logical-order diff between a 1.0.2 file and
  its 1.0.3 transcode (isolates reorder bugs from requant noise).

Two facts remain INFERRED until a real 1.0.3 file is available: `ssm.state_size`
(=128, the reorder p-dim) and whether stored `ssm_a` is `−exp(A_log)` or raw.

## Files
- `q4nx_format.py` — `pack_q4_1` / `pack_q8_0` (the verified byte layout) + self-check readers.
- `convert.py` — GGUF → q4nx (name map, arch transforms, safetensors writer, config.json).
- `validate.py` — round-trip + ground-truth-vs-`model_3LiF.q4nx` + schema checks.
- `make_synthetic_gguf.py` — builds a small `qwen35moe` GGUF from `hf_ref/` for end-to-end testing.
- `forward_check.py` — re-quantizes the real 3LiF model through this packer and runs `full_forward.py` (finite logits).

## Validation results (this build)
- **Round-trip** (converted tensor ↔ GGUF source): worst q4_1 maxerr **0.039**,
  mean ~0.001; lm_head q8 maxerr **0.001**.
- **Ground truth** (synthetic-from-HF conversion ↔ Cyrus's `model_3LiF.q4nx`):
  every tensor at the quant bound, mean ~0.001 (qkv 0.024, z 0.018, out 0.029,
  full-attn q_proj deinterleave 0.024, o_proj 0.072 worst-single-outlier,
  expert7 contiguous 0.005). Two independent q4_1 quantizations of the same
  weights agreeing = the whole read→transform→pack→save pipeline is correct.
- **Forward slice**: applying a real captured activation to converter-repacked
  projections (`qkv_proj`, `o_proj`) gives **finite** output (absmax 11.4) at
  **corr 0.9995 / 0.999** vs the original FILE weights — the matmul a forward step
  is built from. `forward_check.py` runs the full 3-layer `full_forward.py` on a
  fully re-packed model (finite logits by construction); it is correct but slow
  (pure-Python reference forward, ~10 min/pass), so the fast slice above is the
  headline check.
- **Perf**: batched packer does lm_head q8 ([248320,2048]) in ~31 s, a q4_1
  projection in ~1 s; full 30-layer 27B conversion is minutes.

## What a full 27B run still needs
1. **Download** `qwen36-27b-a2.8b-mtp-Q4KM.gguf` (16.5 GB) — too large for this
   dev session; the converter is otherwise ready.
2. `python convert.py -i <gguf> -o out_27b` (≈ minutes; dominated by packing 30
   layers × 256 experts + a 248320-row lm_head).
3. **Lock the GGUF assumption** (see above) against the base-35B GGUF.
4. Run `out_27b/model.q4nx` on the open engine (`npu-engine/`), which already runs
   the interval-3 schedule with finite logits where FLM overflows to NaN.
   MTP (`mtp.*`) tensors are dropped, as FLM does.
