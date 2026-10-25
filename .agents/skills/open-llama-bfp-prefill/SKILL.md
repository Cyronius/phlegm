---
name: open-llama-bfp-prefill
description: Build and validate the experimental compensated-BFP Llama1B GGUF prefill kernels and their explicit C++ input format, including whole-model and paired timing gates.
---

Use the canonical Phlegm checkout for kernels and experiment utilities; the OFLM
runtime reconciliation is a separate source base, not a patch series. Read
`utilities/gguf-prefill/README.md` and the latest continuation in
`specs/open-engine/gguf-prefill-linux.md` there before rebuilding.

The experimental `bfp16_hilo` step tag describes activation packing, not a
weight requantization or the older `bfp16` c-tiled-output route. It requires
T*K*9/4 input bytes, including twin buffers, and produces ordinary F32 `[N,T]`.
The tested fused probe reads6144-byte GGUF-direct Q4 chunks; it cannot be
substituted into the compact5120-byte pool. The fused RHS layout is
`[K64][T32][token8][K8][hi,lo]`; the unfused probe uses token pairs instead.
Host packing must raise the shared exponent when a positive mantissa rounds
to128. Saturating that value silently changes the input.

Build every projection stream belonging to the new image. Use
`build_bfp_stream_probe.py --fused`; `prepare_bfp_hilo_image.py` checks the
three CDO payload hashes before allowing those streams to share one context.
Passing one standalone shape is insufficient. Check long-K down and T>256
input walks too. Keep the probability high/low flash and decode images intact.

Before timing, run the independent source/BF16-operand gate, host encoder and
metadata rejection tests, full300 and8192-position model gates, the4096-boundary
slice, and CLI/bridge logit comparison. Retain the known artificial zero-history
failure separately. Kernel-level tolerance is not permission to skip model
quality. Paired trials exclude compilation and run one NPU job at a time;
coordinate all agents' build completion before timing. Timing-only ablations
produce invalid outputs and must stay labelled as such.

Production defaults remain separate from these experimental manifests. Record
commands, source/artifact hashes, rejected variants and remaining gates in the
active shared plan; contribute new hardware evidence to xdna2-notes/raw/phlegm.
