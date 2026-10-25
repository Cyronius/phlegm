"""Build an isolated compensated-BFP probe; never changes an exported model.

Generates a design from the current canonical GEMM with checked substitutions.
The accepted design's assertions/defaults remain untouched. Compile all streams.
Run numerical checks with check_gemm.py --activation-format bfp-hilo-pairs.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=8192)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--tokens', type=int, default=256)
    ap.add_argument('--fused', action='store_true', help='emit packed panels directly from dequant')
    ap.add_argument('--ablate-matmul', action='store_true', help='TIMING ONLY: wrong outputs')
    ap.add_argument('--ablate-dequant', action='store_true', help='TIMING ONLY: wrong outputs')
    args = ap.parse_args()
    if min(args.n, args.k, args.tokens) <= 0 or any(x % 256 for x in (args.n, args.k, args.tokens)):
        ap.error('dimensions must be positive multiples of256')
    args.out.mkdir(parents=True, exist_ok=False)
    base = ROOT / 'open_kernels/designs/gemm_q4_prefill'
    kernel = base / 'gemm_bfp_compensated_probe.cc'
    s = (base / 'gemm_q4_prefill.py').read_text()
    def swap(old, new):
        nonlocal s
        if s.count(old) != 1:
            raise RuntimeError(f'canonical source changed; review probe substitution: {old[:70]}')
        s = s.replace(old, new)
    swap('HERE = Path(__file__).parent', f'HERE = Path({str(base)!r})')
    swap('assert fmt == "q4_1" or not (bfp16 or wfmt_t2 or xbfp or yt or kt != 64),',
         'assert fmt in ("q4_1", "q4_1_f32") or not (bfp16 or wfmt_t2 or xbfp or yt or kt != 64),')
    a, b = s.index('    nib_ty ='), s.index('    band_ty =')
    s = s[:a] + '''    nib_ty = np.ndarray[(m*k_tile,),np.dtype[dtype_in]]
    scratch_ty = np.ndarray[(m*k_tile*9//4,),np.dtype[np.uint8]]
''' + s[b:]
    swap('arg_types=[band_ty, nib_ty, scratch_ty]', 'arg_types=[band_ty, scratch_ty, nib_ty]')
    a = s.index('    if xbfp:   # the activation')
    b = s.index('    C_l2_ty', a)
    s = s[:a] + '''    X_ty = np.ndarray[(k*t*9//4,),np.dtype[np.uint8]]
    B_l1_ty = np.ndarray[(k_tile*n*9//4,),np.dtype[np.uint8]]
    B_l2_ty = B_l1_ty
''' + s[b:]
    a = s.index('\n', s.index('    C_acc_ty ='))
    s = s[:a] + f'''
    matmul_kernel = ExternalFunction("gqp_comp", source_file={str(kernel)!r},
        arg_types=[scratch_ty,B_l1_ty,C_acc_ty], include_dirs=_include_dirs(), compile_flags=["-DONLY_MM"])
    pack_kernel = ExternalFunction("gqp_pack", source_file={str(kernel)!r},
        arg_types=[nib_ty,scratch_ty], include_dirs=_include_dirs(), compile_flags=["-DONLY_PACK"])
''' + s[a:]
    swap('name=f"A_L2L1_{row}", depth=2', 'name=f"A_L2L1_{row}", depth=1')
    swap('B_l2_mega_ty = np.ndarray[(b_slice_tiles * k_tile * n,), np.dtype[dtype_in]]',
         'B_l2_mega_ty = np.ndarray[(b_slice_tiles*k_tile*n*9//4,),np.dtype[np.uint8]]')
    swap('dequants = rest\n', 'dequants = rest[1:]\n            pack = rest[0]\n')
    swap('dequants[ky](band, local[0], local[1])',
         'dequants[ky](band, local[1], local[0])\n                        pack(local[0],local[1])')
    swap('*dequant_kernels],', 'pack_kernel, *dequant_kernels],')
    a = s.index('    def b_tap(')
    b = s.index('    # ---- C drain taps', a)
    s = s[:a] + '''    def b_tap(col: int, ntile: int) -> TensorAccessPattern:
        block_bytes = TE * 9 // 4
        off = (col + ntile * n_aie_cols) * block_bytes
        return TensorAccessPattern((k*t*9//4,), off, [KB,8,block_bytes//8], [NB*block_bytes,block_bytes//8,1])

''' + s[b:]
    if args.fused:
        # Keep the canonical header unchanged; snapshot its source and replace
        # only the R8 output stores with the separately owned probe emitter.
        h = (base / 'gemm_q4_dequant.h').read_text()
        h = '// Probe adaptation: direct compensated BFP output.\n' + h
        key = '__attribute__((noinline)) inline void gqd_fused_ky'
        assert h.count(key) == 1
        h = h.replace(key, f'#include "{base / "gemm_bfp_compensated_emit.h"}"\n' + key)
        old = 'bfloat16 *__restrict scratch) {\n  aie::set_rounding(aie::rounding_mode::conv_even);\n  AIE_LOOP_RANGE(2, 2)\n  for (unsigned part'
        assert h.count(old) == 1
        h = h.replace(old, 'bfp16ebs8 *__restrict scratch) {\n  aie::set_rounding(aie::rounding_mode::conv_even);\n  aie::block_vector_output_buffer_stream<bfp16ebs8,64> packed(scratch);\n  AIE_LOOP_RANGE(2, 2)\n  for (unsigned part')
        old = """        bfloat16 *__restrict dst = scratch + sub * (8 * GQD_S);
        const unsigned ib8 = ib0 / 2;
        aie::store_v(dst + (ib8 + 0) * GQD_COLA * (8 * GQD_S), z.first);
        aie::store_v(dst + (ib8 + 1) * GQD_COLA * (8 * GQD_S), z.second);"""
        assert h.count(old) == 1
        h = h.replace(old, '        gqp_probe_emit(z.first,packed);\n        gqp_probe_emit(z.second,packed);')
        old = '    bfloat16 *__restrict scratch) {\n  event0();'
        assert h.count(old) == 1
        h = h.replace(old, '    bfp16ebs8 *__restrict scratch) {\n  event0();')
        header = (args.out / 'dequant.h').resolve()
        header.write_text(h)
        entries = []
        for ky in range(4):
            entry = (args.out / f'entry{ky}.cc').resolve()
            entry.write_text(f'#include "{header}"\nextern "C" void gqd_entry_k{ky}(const uint8_t *a,uint8_t *b,bfp16ebs8 *c) {{gqd_dequant_ky(a,{ky},b,c);}}\n')
            entries.append(str(entry))
        swap('nib_ty = np.ndarray[(m*k_tile,),np.dtype[dtype_in]]', 'nib_ty = np.ndarray[(64,),np.dtype[np.uint8]]')
        swap('arg_types=[band_ty, scratch_ty, nib_ty]', 'arg_types=[band_ty, nib_ty, scratch_ty]')
        swap('dequant_srcs = ensure_dequant_entries(bfp=xbfp)', f'dequant_srcs = {entries!r}')
        swap('compile_flags=["-DONLY_MM"]', 'compile_flags=["-DONLY_MM","-DROW_PAIRS"]')
        a = s.index('    pack_kernel = ExternalFunction(')
        b = s.index('\n\n', a)
        s = s[:a] + s[b:]
        swap('dequants = rest[1:]\n            pack = rest[0]', 'dequants = rest')
        swap('dequants[ky](band, local[1], local[0])\n                        pack(local[0],local[1])', 'dequants[ky](band, local[0], local[1])')
        swap('pack_kernel, *dequant_kernels],', '*dequant_kernels],')
        swap('name=f"A_L2L1_{row}", depth=1', 'name=f"A_L2L1_{row}", depth=2')
    design = args.out / 'design.py'
    design.write_text(s)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GQP_')}
    env.update(GQP_N=str(args.n), GQP_K=str(args.k), GQP_T=str(args.tokens),
               GQP_SCALES_F32='1', GQP_BFP16='1', OFLM_KEEP_FAILED='1')
    if args.ablate_matmul:
        env['GQP_NULL_MM'] = '1'
    if args.ablate_dequant:
        env['GQP_NULL_DEQUANT'] = '1'
    if args.ablate_matmul or args.ablate_dequant:
        print('TIMING ONLY ABLATION: outputs are invalid', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'open_kernels/build_design.py'),
                    str(design), str(args.out / 'build')], env=env, check=True)


if __name__ == '__main__':
    main()
