r"""MoE combine on the NPU (one core):
  out = xres + sum_e w[e]*y_e + sigmoid(xm . sgw) * shared

Args: rout f32[1024] (router output: w[8] at 264..271), y f32[8*2048] (expert down
outputs, expert-major), xres f32[2048], shared f32[2048], xm bf16[2048],
sgw bf16[2048], out f32[2048].
Streams: in = 4 KB elements [rout][y_e (2) x 8][xres (2)][shared (2)][xm][sgw]; out = 2 elements.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N = 2048
NE = 8


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def moe_combine(rout: In, y: In, xres: In, shared: In, xm: In, sgw: In, out: Out, *, srchash: CompileTime[int] = 0):
    u8 = np.ndarray[(4096,), np.dtype[np.uint8]]
    f8 = np.ndarray[(8,), np.dtype[np.float32]]
    facc = np.ndarray[(N,), np.dtype[np.float32]]
    r_ty = np.ndarray[(1024,), np.dtype[np.float32]]
    y_ty = np.ndarray[(NE * N,), np.dtype[np.float32]]
    f_ty = np.ndarray[(N,), np.dtype[np.float32]]
    b_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    inc = include_dirs()
    f_w = ExternalFunction("mc_wcopy", source_file=str(HERE / "mc_wcopy.cc"), arg_types=[u8, f8], include_dirs=inc)
    f_a = ExternalFunction("mc_axpy", source_file=str(HERE / "mc_axpy.cc"), arg_types=[facc, u8, u8, f8, np.int32], include_dirs=inc)
    f_f = ExternalFunction("mc_fin", source_file=str(HERE / "mc_fin.cc"), arg_types=[facc, u8, u8, u8, u8, u8, u8, u8, u8], include_dirs=inc)
    of_in = ObjectFifo(u8, name="in", depth=6)
    of_out = ObjectFifo(u8, name="out", depth=2)
    wb = Buffer(f8, name="wb")
    acc = Buffer(facc, name="acc")

    def core_body(ain, aout, wb, acc, fw, fa, ff):
        e = ain.acquire(1)
        fw(e, wb)
        ain.release(1)
        for i in range_(NE):
            e = ain.acquire(2)
            fa(acc, e[0], e[1], wb, i)
            ain.release(2)
        e = ain.acquire(6)
        o = aout.acquire(2)
        ff(acc, e[0], e[1], e[2], e[3], e[4], e[5], o[0], o[1])
        aout.release(2)
        ain.release(6)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), wb, acc, f_w, f_a, f_f],
                    tile=Tile(0, 2), stack_size=0x1800)

    def sequence(a_r, a_y, a_x, a_s, a_xm, a_sgw, c_out, inp, outc):
        pipe = Pipeline(3)
        pipe.drain(outc, c_out, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_r, TensorAccessPattern((1, 1024), 0, [1, 1, 1, 1024], [0, 0, 0, 1]))
        pipe.fill(inp, a_y, TensorAccessPattern((1, NE * N), 0, [1, 1, 1, NE * N], [0, 0, 0, 1]))
        pipe.fill(inp, a_x, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_s, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_xm, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.fill(inp, a_sgw, TensorAccessPattern((1, N), 0, [1, 1, 1, N], [0, 0, 0, 1]))
        pipe.finish()

    rt = Runtime(sequence, [r_ty, y_ty, f_ty, f_ty, b_ty, b_ty, f_ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


DESIGN = moe_combine
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h")) + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
