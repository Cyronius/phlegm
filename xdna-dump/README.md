# xdna-dump

What we know about the AMD XDNA2 NPU (the AIE2P array in Strix Point, Strix Halo, Krackan and Gorgon Point), its
toolchain and its firmware, plus what we found in FastFlowLM along the way. The website will publish it at `/xdna`.

## Layout

| dir | content |
|---|---|
| `sources/` | Raw inputs: phlegm's old `docs/` and the technical plans from its archive. These are kept as written, with no edits. |
| `tools/` | The tools that produced the findings: `seq-capture/` (the proxy `xrt_coreutil.dll` that recorded FastFlowLM's NPU traffic), `kernel-interp/` (the Python CPU replicas and pool builders), `golden-ref/` (the HF reference forward), `q4nx/` (q4nx format validation). |

The synthesized pages will go here as `hardware/`, `toolchain/`, `traps.md`, `formats/`, `flm-findings/` and `perf/`.
Other source material for them:

- `../open_kernels/README.md`: the kernel status table and the trap catalogue.
- `../open_kernels/designs/*/README.md` and `NOTES.md`.
- `../specs/open-engine/spec.md` and its plans.

## Rules

- Every claim has a date and a source: a commit, a file or a capture.
- When a later finding contradicts an earlier claim, rewrite the page. Move the old claim to a "superseded" footnote
  with its date. Do not append the correction under the old text.
- Commits straight to `main` are allowed for `xdna-dump/**` only. Everything else goes through a PR.
- `sources/` is input. Do not edit it; write the synthesized page instead.
