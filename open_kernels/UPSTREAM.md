# Sync ledger: phlegm <-> openflowlm-next

phlegm is the canonical home of `open_kernels/` and `specs/open-engine/`. openflowlm-next
(https://github.com/Atomic-Germ/openflowlm-next) receives the work as pull requests. Both trees use the same paths, so a
sync is a plain diff of these two directories.

## Rules

- Keep the manifest contract compatible. A change to the manifest schema, its `manifest_version`, or a recipe function's
  signature goes upstream together with the matching change to openflowlm-next's `src/open_qwen36` reader.
- Open one upstream PR per finished feature. Do not hold work for a large batch.
- Bring other contributors' commits in with `git cherry-pick -x`, so that their authorship stays.
- When review changes one of our PRs upstream, pull the merged version back in.

## Pulled in from openflowlm-next

| date | what | openflowlm-next ref |
|---|---|---|
| 2026-10-19 | `main`, full history of both dirs | `c05b00fd66fe16f2cbb03f2894c49cf5aea61d4a` |
| 2026-10-19 | PR #209 perf/granite-speedup (open, not merged upstream) | `814640b` |
| 2026-10-19 | PR #184 feat/k2-horizon-7b (open, not merged upstream) | `03a562e` |
| 2026-10-19 | PR #203 feat/bonsai2-dflash2-draft (open, not merged upstream) | `0cea11f` |

## Not pulled in yet

| PR | why |
|---|---|
| #193 perf/dense-prefill-gemm8 | It takes `manifest_version` 4 for `bfp16_dit` (`DIT_MANIFEST_VERSION`). #209 already uses 4, 5 and 6 (`BFP16`, `GLU`, `PREP`). The numbers must be agreed upstream first, in the C++ reader too. |
| #194 perf/k2-batched-attn | It conflicts with #209 in `recipes/dense.py` and in the spec. Import it after #193, or after either PR lands upstream. |

## Changes made here that upstream does not have

| date | change |
|---|---|
| 2026-10-19 | #203's `OPEN-DECODE-ONE-CONTEXT-DENSE` (qwen35 `dux.py`) collided with #209's requirement of the same ID (dense `dx.py`). In phlegm, #203's requirement is `OPEN-DECODE-ONE-CONTEXT-QWEN35`. Upstream, one of the two PRs must rename it the same way. |
| 2026-10-19 | The union of #209 and #203: both `CLEAR` knob lists in `export_qwen36_kernels.py`, and both the `rows` and `verify` manifest keys. |
| 2026-10-19 | `designs/gemm_q4/NOTES.md`: the gemm_q4 working notes from phlegm's old README. |

## Pushed out to openflowlm-next

| date | phlegm ref | upstream PR | state |
|---|---|---|---|
