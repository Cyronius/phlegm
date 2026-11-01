r"""dxm: Gemma 4 26B-A4B's layer as ONE image a layer type -- dx.py's attention + dense MLP and gm.py's
routed experts on the same xclbin, as two instruction streams (OPEN-FAMILY-GEMMA4-MOE; R-0003).

Measured 2026-11-01 (target/npu-dev/gemma4-26b-perf): with dx and gm_moe on separate contexts, every
layer changes context twice. dx_local alone in its context takes 1.08-1.32 ms, but 1.89-1.92 ms
inside the real walk; gm_moe (4 cores) takes 1.11 ms either way, the 8-core gm8 0.73-0.90 alone
against 1.17 in the walk. Entering an image costs with the core programs it loads, so the experts
join dx's image instead of the other way round.

dx.py's image uses main cores Tile(0..3, 2), the norm helper Tile(0, 3), the attention cores on row
4, every MM2S channel of shim columns 0-3 and one of each of columns 4-6 (the per-core KV windows).
That leaves Tile(4..7, 2) and exactly five MM2S channels (columns 4-6 ch 1, column 7 both), which is
gm.py's 4-core expert kernel (4 weight streams + xm2), not gm8's nine. gm.py's cores go there
unchanged (gm.parts(col0=4, x_col=7)); its outputs drain on columns 4-7, whose S2MM are free.

G4_DXM_STREAM selects which runtime sequence this build emits; the image is the same either way
(every fifo of both is in the Runtime, as DX_HEAD's head stream does). Only the dx build's xclbin is
loaded, so the two builds' xclbins must agree apart from build stamps (export_qwen36_kernels.
xclbin_equivalent; target/npu-dev/gemma4-26b-build.sh runs it):
    dx  -- dx.py's layer sequence (the stream `dx_local` / `dx` run)
    gm  -- gm.py's expert sequence over (pool, act) = the image's args 0 and 4 (`gm_l` / `gm_g`)

dx.py is not edited: this module loads it and hands it a Runtime / Program that add gm's endpoints,
workers and sequence. The layer's act grows to the MoE layout's (gm's regions sit past dx's own).

Build: OPEN_KERNELS_SPEC=<26B spec> G4_KIND=local|global G4_DX=1 GEMV_Q4_0=1 G4_DXM_STREAM=dx|gm \
       python build_design.py designs/gemma4_moe/dxm.py <out>
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DENSE = HERE.parent / "dense"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

STREAM = os.environ.get("G4_DXM_STREAM", "dx")
if STREAM not in ("dx", "gm"):
    raise SystemExit(f"dxm: G4_DXM_STREAM={STREAM!r} (dx | gm)")
GM_COL0, GM_XCOL = 4, 7


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GM = _load("g4_gm_parts", HERE / "gm.py")
DX = _load("g4_dx_layer", DENSE / "dx.py")
_ADDED: dict = {}
_Runtime, _Program = DX.Runtime, DX.Program


def _runtime(sequence, fn_args):
    """dx.py's Runtime, with gm's endpoints after its own and the stream this build emits"""
    workers, gm_args, gm_seq = GM.parts(col0=GM_COL0, x_col=GM_XCOL)
    _ADDED["workers"] = workers
    n = len(fn_args)
    io = list(fn_args)
    u8 = np.dtype[np.uint8]
    io[0] = np.ndarray[(max(GM.L_POOL, DX.L.POOL_BYTES),), u8]      # the layer pool: experts + dense
    io[4] = np.ndarray[(max(GM.M.AD_BYTES, DX.L.AD_BYTES),), u8]    # act: dx's regions, then gm's

    def seq(*a):
        if STREAM == "gm":
            gm_seq(a[0], a[4], *a[n:])
        else:
            sequence(*a[:n])
    return _Runtime(seq, io + gm_args)


def _program(device, rt, workers):
    return _Program(device, rt, workers=list(workers) + _ADDED.pop("workers"))


assert not DX.HEAD and not DX.QKVB, "dxm: the layer stream only"
assert all(GM_COL0 + c >= DX.N_CORES for c in range(GM.N_CORES)) and GM_COL0 + GM.N_CORES <= 8
DX.Runtime, DX.Program = _runtime, _program

DESIGN = DX.DESIGN
_src = (b"".join(sorted(f.read_bytes() for f in HERE.glob("*.py"))) + STREAM.encode()
        + f"{GM_COL0},{GM_XCOL}".encode())
SPECIALIZE = dict(DX.SPECIALIZE, srchash=int(hashlib.sha1(_src + str(DX.SPECIALIZE["srchash"]).encode())
                                             .hexdigest()[:8], 16))
