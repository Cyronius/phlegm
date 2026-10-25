"""Build one experimental compensated64 stream; generated sources stay under target/."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'PLAN.md').exists() and (p / 'open_kernels').is_dir())
CDOS = ['main_aie_cdo_elfs.bin', 'main_aie_cdo_init.bin', 'main_aie_cdo_enable.bin']


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def fused_header(source):
    def swap(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f'canonical dequant changed; review substitution: {old[:70]}')
        source = source.replace(old, new)
    key = '__attribute__((noinline)) inline void gqd_fused_ky'
    swap(key, f'#include "{HERE.parent / "gemm_bfp_compensated_emit.h"}"\n' + key)
    swap('bfloat16 *__restrict scratch) {\n  aie::set_rounding(aie::rounding_mode::conv_even);\n  AIE_LOOP_RANGE(2, 2)\n  for (unsigned part',
         'bfp16ebs8 *__restrict scratch) {\n  aie::set_rounding(aie::rounding_mode::conv_even);\n  aie::block_vector_output_buffer_stream<bfp16ebs8,64> packed(scratch);\n  AIE_LOOP_RANGE(2, 2)\n  for (unsigned part')
    swap('''        bfloat16 *__restrict dst = scratch + sub * (8 * GQD_S);
        const unsigned ib8 = ib0 / 2;
        aie::store_v(dst + (ib8 + 0) * GQD_COLA * (8 * GQD_S), z.first);
        aie::store_v(dst + (ib8 + 1) * GQD_COLA * (8 * GQD_S), z.second);''',
         '        gqp_probe_emit(z.first,packed);\n        gqp_probe_emit(z.second,packed);')
    swap('    bfloat16 *__restrict scratch) {\n  event0();',
         '    bfp16ebs8 *__restrict scratch) {\n  event0();')
    return '// Generated compensated64 adaptation: packed BFP high/residual output.\n' + source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--n', type=int, default=8192)
    ap.add_argument('--k', type=int, default=2048)
    ap.add_argument('--compare-cdo', type=Path, help='existing build directory; reject a different hardware image')
    args = ap.parse_args()
    if args.n <= 0 or args.n % 512 or args.k <= 0 or args.k % 256:
        ap.error('N must be a positive multiple of512 and K a positive multiple of256')
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'target'):
        ap.error('generated probes must stay under the ignored target directory')
    out.mkdir(parents=True, exist_ok=False)
    generated = out / 'source'
    generated.mkdir()
    parent = HERE.parent / 'gemm_q4_dequant.h'
    watched = [parent, HERE.parent / 'gemm_bfp_compensated_emit.h', HERE / 'design.py', HERE / 'mm.cc', Path(__file__)]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in watched}
    header = generated / 'dequant.h'
    header.write_text(fused_header(parent.read_text()))
    for ky in range(4):
        (generated / f'entry{ky}.cc').write_text(
            f'#include "{header}"\nextern "C" void gqd_entry_k{ky}(const uint8_t *a,uint8_t *b,bfp16ebs8 *c) {{gqd_dequant_ky(a,{ky},b,c);}}\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GQP_', 'COMPENSATED64_'))}
    env.update(COMPENSATED64_SOURCE_DIR=str(generated), COMPENSATED64_N=str(args.n), COMPENSATED64_K=str(args.k), OFLM_KEEP_FAILED='1')
    command = [sys.executable, str(ROOT / 'open_kernels/build_design.py'), str(HERE / 'design.py'), str(out / 'build')]
    with (out / 'build.log').open('w') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for p in watched:
        if digest(p) != hashes[str(p.relative_to(ROOT))]:
            raise RuntimeError(f'source changed during compilation: {p}')
    cdo = {name: digest(out / 'build/final.prj/cdo_main' / name) for name in CDOS}
    comparison = None
    if args.compare_cdo:
        previous = {name: digest(args.compare_cdo / 'final.prj/cdo_main' / name) for name in CDOS}
        comparison = dict(previous=str(args.compare_cdo), cdo=previous, identical=previous == cdo)
    metadata = dict(kind='experimental-compensated64', n=args.n, k=args.k, tokens=256,
                    weight_chunk_bytes=6144, output='float32[N,T]', input='bfp16_hilo',
                    source_sha256=hashes, cdo_sha256=cdo, comparison=comparison,
                    artifacts={name: digest(out / 'build' / name) for name in ['final.xclbin', 'insts.bin', 'insts.elf']},
                    hardware_validation='not performed by builder')
    (out / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if comparison is not None and not comparison['identical']:
        raise RuntimeError('CDO payload differs; inspect build.json, do not mix streams')
    print(json.dumps(dict(out=str(out), cdo_identical=None if comparison is None else comparison['identical']), indent=2))


if __name__ == '__main__':
    main()
