# NPU sequence capture (Tier 0)

Capture the *live* per-op control sequences the closed FLM engines submit to the
NPU, in submission order, **without recompiling the engine and without Ghidra** —
then diff an interval-4 run against an interval-3 run to locate the interval-3
scheduling defect (see `docs/rust-engine-replacement-feasibility.md`).

## Why this works

`qwen3_6_moe_npu.dll` (and every other engine DLL) dynamically imports exactly
one non-system DLL that touches the NPU: **`xrt_coreutil.dll`** (verified from the
engine's PE import table). Every control-code blob the engine builds passes
through `xrt::elf::elf(const void* buf, size_t n)` before it is turned into a
module and submitted. That is the choke point.

A header hook on `npu_app` (npu_utils_xrt.hpp) does **not** work: the engine is a
prebuilt DLL with its own baked-in copy of that template, so editing the header
only affects `flm.exe`, never the engine's own submissions.

The three moving parts:

| File | Role |
|---|---|
| `gen_def.py` | Parses the real `xrt_coreutil.dll` export table (pure stdlib) and regenerates `xrt_coreutil.def`. |
| `xrt_coreutil.def` | Forwards all 541 exports to the renamed real DLL. Already generated. |
| `xrt_shim.cpp` | The proxy. Forwards everything; Detour-hooks the elf ctor (dump each blob), `run::start`, `run::wait`. |
| `seq_diff.py` | Byte-level diff of two corpora. Noisy: weight-offset patching makes most ops byte-unique. |
| `seq_struct.py` | **Use this.** Parses the AIE ctrlcode, fingerprints each op by structure with weight pointers masked, and diffs op *types*. Collapses hundreds of byte-signatures to ~16 real op types. |
| `run_capture.ps1` | Drives one server run: start → one prompt → stop. |

## Build (Windows, x64)

Needs MSVC + Microsoft Detours (`vcpkg install detours:x64-windows`).

```pwsh
cd tools/seq-capture
cmake -B build -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
# -> build/out/xrt_coreutil.dll   (the proxy)
```

If the real DLL on your machine ever changes version, regenerate the .def first:

```pwsh
python gen_def.py C:\Windows\System32\xrt_coreutil.dll   # rewrites xrt_coreutil.def
```

## Install the proxy (this machine, done)

FLM loads `xrt_coreutil.dll` from `C:\Windows\System32`. To interpose, put both
files next to `flm.exe` (the app dir wins the DLL search) — additive, reversible:
```pwsh
copy C:\Windows\System32\xrt_coreutil.dll C:\flm-test\xrt_coreutil_orig.dll
copy build\out\xrt_coreutil.dll           C:\flm-test\xrt_coreutil.dll
```
When `FLM_SEQ_CAPTURE_DIR` is unset the proxy is a pure pass-through, so it is
safe to leave installed (verified: `flm validate` still detects the NPU).

## Run a capture (verified protocol)

Env that matters (learned the hard way):
- `FLM_MODEL_PATH=C:\Users\josha\.flm`  ← the `.flm` root, NOT `.flm\models`
  (FLM appends `models\` itself; the deeper path triggers a re-download).
- `FLM_SEQ_CAPTURE_DIR=<dir>` arms the capture.
- Server: `flm serve qwen3.6-moe:35b-a3b --port 52625 --quiet`; chat endpoint is
  OpenAI-style `POST http://127.0.0.1:52625/v1/chat/completions`.

The repro pair lives in the model dir as `model_8Li4.q4nx`/`config_8Li4.json`
(interval-4, works) and `model_6Li3.q4nx`/`config_6Li3.json` (interval-3, broken).
FLM loads fixed `model.q4nx`/`config.json`, so swap the variant into those names
(originals are backed up as `*.orig`), run one prompt, stop, repeat. `run_capture.ps1`
drives one run end to end. Use the **same prompt and `max_tokens`** for both.

Then analyse (prefer the structural view):
```pwsh
python seq_struct.py C:\caps\i4 C:\caps\i3   # op-type diff, weight-offset view
python seq_diff.py   C:\caps\i4 C:\caps\i3   # raw-byte diff (noisy)
```

Each capture dir gets `NNNNNN.seq` (raw ctrlcode blobs) + `trace.tsv`
(idx, kind, size, fnv1a). `seq_diff.py` labels each distinct op signature
(A, B, C, …) and reports where the two ordered op streams diverge — a missing,
extra, or reordered layer op is the interval-3 defect made visible.

When `FLM_SEQ_CAPTURE_DIR` is unset the proxy is a pure pass-through, so it is
safe to leave installed.

## Reading the result

- **Op-count table** — if the broken run is missing a full-attention op (or has
  the wrong ratio of DeltaNet:full-attn ops per block), that is the scheduler
  mis-mapping `full_attention_interval=3`.
- **Ordered alignment** — the first `[delete]/[insert]/[replace]` region is where
  the engine's layer schedule first departs from the correct `[L,L,F]` cadence.
- The dumped `.seq` blobs are AIE ctrlcode and can be disassembled with the
  in-repo tooling (`npu_sequence::from_file(...).interpret()` in
  `npu_instr_utils.hpp`) for byte-level confirmation.

## Tensor-data capture (Tier-0b) — the plane that found the bug

The op sequence turned out identical between interval-3 and interval-4, so the
defect is in the tensor DATA the CPU computes, not the schedule. The proxy also
hooks the engine's three `xrt::bo` data-path imports:

| Symbol | Role |
|---|---|
| `?map@bo@xrt@@QEAAPEAXXZ` | `void* bo::map()` — records `this → host_ptr`. |
| `?sync@bo@xrt@@QEAAXW4…@_K1@Z` | `bo::sync(dir,size,offset)` — dumps/hashes the exact bytes crossing the boundary (H2D before the flush, D2H after). |
| `??1bo@xrt@@QEAA@XZ` | `~bo()` — evicts the `this` entry so a recycled address can't alias a dead buffer. |

There is **no `bo::write` import**, so map + sync is the entire path. Arm with
`FLM_BO_CAPTURE_DIR`; `FLM_BO_DUMP_MAX` (bytes) caps how much of each sync is
written to a `NNNNNN.bo` file (0 = metadata only). Output: `bo_trace.tsv` with
`idx, dir(H2D|D2H), size, offset, fnv1a, dumped`.

```pwsh
# variant swap + one run (FORWARD-SLASH the capture dir; bash eats backslashes)
pwsh -File bo_capture.ps1 -Variant 6Li3 -CaptureDir C:/caps/bo_i3 -DumpMax 1048576 -MaxTokens 4
python analyze_bo.py C:/caps/bo_i4_meta C:/caps/bo_i3_meta   # decode-loop hash-lock view
python scan_nan.py  C:/caps/bo_i3_dump  C:/caps/bo_i4_dump   # first-NaN localization
```

**Result:** the NPU stages every tensor through fixed 1 MB DMA tiles; the fp32
logits (vocab 248320 ≈ 970 KB) fit one tile and are identifiable. Interval-3's
first logits (end of prefill) are **100 % NaN**; every decode step then returns
the same NaN tile → argmax pinned → `////////`. Interval-4 logits are finite with
zero NaN and a per-step-varying argmax. The blowup is a numeric overflow in the
interval-3 prefill forward pass — not scheduling. See the plan doc's Tier-0b.

## Scope / caveats

- Capturing your own machine's runs for interop debugging is fine. The dumped
  blobs are derived from the closed engine — keep them local; do not redistribute
  a captured corpus (see `src/xclbins/.../TERMS.md`).
- This locates the bug; it does not fix it. The fix is either an AMD/FLM engine
  patch (feed them this diff) or the Tier-1 replay engine.
