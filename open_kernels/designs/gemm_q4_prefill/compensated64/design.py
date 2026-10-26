"""Experimental Q4 GGUF prefill:8 feature columns x4 token rows,64x64 FP32 C/core.

Weights retain their6144-byte chunk ABI. The fused dequant emits BFP high/low
panels; the leaf MAC accumulates HH+HL+LH in FP32. Only T256 is supported.
Run build.py to generate the checked dequant header outside the source tree.
"""
from pathlib import Path
import os

import numpy as np
import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker, kernels, str_to_dtype
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).resolve().parent
T, M, KT, TN = 256, 64, 64, 64
FEATURE_COLS, TOKEN_ROWS = 8, 4
BAND_K, BAND_BYTES = 256, 12288
SOURCE = Path(os.environ['COMPENSATED64_SOURCE_DIR'])
WEIGHT_DEPTH = int(os.environ.get('COMPENSATED64_WEIGHT_DEPTH', '1'))
HIGH_B = os.environ.get('COMPENSATED64_HIGH_B') == '1'
HIGH_ONLY = os.environ.get('COMPENSATED64_HIGH_ONLY') == '1'


def include_dirs():
    from aie.iron.kernels._common import _detect_arch, _include_dirs
    from aie.utils import config
    inc = _include_dirs()
    root = Path(config.cxx_header_path()) / 'aie_kernels'
    return inc + [str(root), str(root / _detect_arch())]


@iron.jit(aiecc_flags=['--alloc-scheme=basic-sequential'])
def gemm_q4_prefill(W: In, X: In, Y: Out, *, n_weight: CompileTime[int], k: CompileTime[int]):
    assert n_weight > 0 and n_weight % 512 == 0, 'N must be a positive multiple of512'
    assert k > 0 and k % 256 == 0, 'K must be a positive multiple of256'
    nrb, nbg, nkt = n_weight // 512, k // BAND_K, k // KT
    t_tiles = T // (TN * TOKEN_ROWS)
    dtype_in, dtype_out = str_to_dtype('bf16'), str_to_dtype('f32')

    # Preserve the tested SDK zero kernel; the SDK matmul itself is not used.
    sdk_mm = kernels.mm(dim_m=M, dim_k=KT, dim_n=TN, input_dtype=dtype_in,
                        output_dtype=dtype_out, b_col_maj=False, c_col_maj=False,
                        use_chess=False, emulate_bf16_mmul_with_bfp16=True, vectorized=True)
    zero = sdk_mm.zero
    r, s, mt = sdk_mm.mac_dims
    assert (r, s, mt) == (8, 8, 8)

    band_ty = np.ndarray[(BAND_BYTES,), np.dtype[np.uint8]]
    nib_ty = np.ndarray[(64,), np.dtype[np.uint8]]
    a_ty = np.ndarray[(M * KT * 9 // (8 if HIGH_ONLY else 4),), np.dtype[np.uint8]]
    b_ty = np.ndarray[(KT * TN * 9 // (8 if HIGH_B else 4),), np.dtype[np.uint8]]
    c_ty = np.ndarray[(M, TN), np.dtype[dtype_out]]
    c_flat_ty = np.ndarray[(M * TN,), np.dtype[dtype_out]]
    c_join_ty = np.ndarray[(M * TN * TOKEN_ROWS,), np.dtype[dtype_out]]
    w_ty = np.ndarray[(n_weight * k * 3 // 4,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(k * T * 9 // 4,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(n_weight * T,), np.dtype[dtype_out]]

    dequants = [ExternalFunction(f'gqd_entry_k{ky}', source_file=str(SOURCE / f'entry{ky}.cc'),
                                arg_types=[band_ty, nib_ty, a_ty], include_dirs=include_dirs(),
                                compile_flags=['-DGQD_R8', '-DGQD_SCALES_F32=1']) for ky in range(4)]
    mm = ExternalFunction('gqp_comp', source_file=str(HERE / ('mm_high_only.cc' if HIGH_ONLY else 'mm.cc')),
                          arg_types=[a_ty, b_ty, c_flat_ty], include_dirs=include_dirs(),
                          compile_flags=['-DCOMPENSATED64_HIGH_B=1'] if HIGH_B else
                          ['-DCOMPENSATED64_TURBO_HH=1'] if os.environ.get('COMPENSATED64_TURBO_HH') == '1' else [])

    # One weight stream per feature column, broadcast to its four token rows.
    a_l3 = [ObjectFifo(band_ty, name=f'A_L3L2_{f}', depth=2) for f in range(FEATURE_COLS)]
    a_l1 = [a_l3[f].cons().forward(obj_type=band_ty, name=f'A_L2L1_{f}',
                                  depth=WEIGHT_DEPTH, tile=Tile(f, 1)) for f in range(FEATURE_COLS)]
    # One activation stream per token row, broadcast across eight feature columns.
    b_l3, b_l1 = [], []
    for row in range(TOKEN_ROWS):
        fifo = ObjectFifo(b_ty, name=f'B_L3L2_{row}', depth=2)
        b_l3.append(fifo)
        b_l1.append(fifo.cons().forward(obj_type=b_ty, name=f'B_L2L1_{row}',
                                        depth=2, tile=Tile(2 * row, 1)))
    # Each physical feature column joins four token-row outputs, never eight.
    c_l3 = [None] * FEATURE_COLS
    c_l1 = [[None] * TOKEN_ROWS for _ in range(FEATURE_COLS)]
    for f in range(FEATURE_COLS):
        dims = [(M // r, r * TN), (r, mt), (TN // mt, r * mt), (mt, 1)]
        fifo = ObjectFifo(c_join_ty, name=f'C_L2L3_{f}', depth=2, dims_to_stream=dims)
        c_l3[f] = fifo
        parts = fifo.prod().join([M * TN * i for i in range(TOKEN_ROWS)],
                                 obj_types=[c_ty] * TOKEN_ROWS,
                                 names=[f'C_L1L2_{f}_{row}' for row in range(TOKEN_ROWS)],
                                 depths=[1] * TOKEN_ROWS, tile=Tile(f, 1))
        for row in range(TOKEN_ROWS):
            c_l1[f][row] = parts[row]

    def grid(ty, name):
        return [[Buffer(ty, name=f'{name}_{f}_{row}') for row in range(TOKEN_ROWS)] for f in range(FEATURE_COLS)]

    nib, packed_a = grid(nib_ty, 'nib'), grid(a_ty, 'ascr')
    rtp = [[Buffer(np.ndarray[(2,), np.dtype[np.int32]], name=f'rtp_{f}_{row}',
                   initial_value=np.zeros(2, dtype=np.int32), use_write_rtp=True)
            for row in range(TOKEN_ROWS)] for f in range(FEATURE_COLS)]

    def core_fn(in_a, in_b, out_c, zero, matmul, nibble, a, my_rtp, *dq):
        for _ in range_(t_tiles):
            c = out_c.acquire(1)
            zero(c)

            def one_band(band):
                for ky in range(4):
                    dq[ky](band, nibble, a)
                    b = in_b.acquire(1)
                    matmul(a, b, c)
                    in_b.release(1)

            # The first fill follows RTP writes; acquire before reading the count.
            band = in_a.acquire(1)
            count = my_rtp[1]
            one_band(band)
            in_a.release(1)
            for _ in range_(count - 1):
                band = in_a.acquire(1)
                one_band(band)
                in_a.release(1)
            out_c.release(1)

    def make_worker(f, row):
        return Worker(core_fn,
                      [a_l1[f].cons(), b_l1[row].cons(), c_l1[f][row].prod(),
                       zero, mm, nib[f][row], packed_a[f][row], rtp[f][row], *dequants],
                      tile=Tile(f, 2 + row), stack_size=0x1000)

    workers = Worker.grid(FEATURE_COLS, TOKEN_ROWS, make_worker)
    row_bytes = nbg * BAND_BYTES
    block_bytes = KT * TN * 9 // 4
    nb = T // TN

    def a_tap(f, rbg):
        offset = f * row_bytes + rbg * FEATURE_COLS * row_bytes
        return TensorAccessPattern((n_weight * k * 3 // 4,), offset,
                                   [1, row_bytes], [row_bytes, 1])

    def b_tap(row, ntile):
        offset = (row + ntile * TOKEN_ROWS) * block_bytes
        if HIGH_B:
            # Gather only high72B panels from unchanged hi/lo DDR storage.
            if nkt <= 64:
                sizes, strides = [nkt, 64, 72], [nb * block_bytes, 144, 1]
            else:
                assert nkt % 64 == 0 and nkt // 64 <= 64
                sizes = [nkt // 64, 64, 64, 72]
                strides = [64 * nb * block_bytes, nb * block_bytes, 144, 1]
            return TensorAccessPattern((k * T * 9 // 4,), offset, sizes, strides)
        return TensorAccessPattern((k * T * 9 // 4,), offset, [nkt, 8, block_bytes // 8],
                                   [nb * block_bytes, block_bytes // 8, 1])

    def c_tap(rbg, ntile, f):
        offset = (rbg * FEATURE_COLS + f) * M * T + ntile * TOKEN_ROWS * TN
        return TensorAccessPattern((n_weight, T), offset, [TOKEN_ROWS, M, TN], [TN, T, 1])

    a_prods, b_prods, c_conses = [f.prod() for f in a_l3], [f.prod() for f in b_l3], [f.cons() for f in c_l3]

    def sequence(a_w, a_x, c_y, aw, bx, cy):
        for f in range(FEATURE_COLS):
            for row in range(TOKEN_ROWS):
                rtp[f][row][0] = t_tiles
                rtp[f][row][1] = nbg
        for rbg in range(nrb):
            for ntile in range(t_tiles):
                tg = TaskGroup()
                for f in range(FEATURE_COLS):
                    aw[f].fill(a_w, tap=a_tap(f, rbg), group=tg)
                for row in range(TOKEN_ROWS):
                    bx[row].fill(a_x, tap=b_tap(row, ntile), group=tg)
                for f in range(FEATURE_COLS):
                    cy[f].drain(c_y, tap=c_tap(rbg, ntile, f), wait=True, group=tg)
                tg.finish()

    rt = Runtime(sequence, [w_ty, x_ty, y_ty, a_prods, b_prods, c_conses])
    program = Program(iron.get_current_device(), rt, workers=[w for feature in workers for w in feature])
    return program.resolve_program()


DESIGN = gemm_q4_prefill
SPECIALIZE = {'n_weight': int(os.environ['COMPENSATED64_N']), 'k': int(os.environ['COMPENSATED64_K'])}
