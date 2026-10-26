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


def fused_header(source, high_only=False):
    def swap(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f'canonical dequant changed; review substitution: {old[:70]}')
        source = source.replace(old, new)
    key = '__attribute__((noinline)) inline void gqd_fused_ky'
    emitter = HERE / 'high_only_emit.h' if high_only else HERE.parent / 'gemm_bfp_compensated_emit.h'
    swap(key, f'#include "{emitter}"\n' + key)
    swap('bfloat16 *__restrict scratch) {\n  aie::set_rounding(aie::rounding_mode::conv_even);\n  AIE_LOOP_RANGE(2, 2)\n  for (unsigned part',
         'bfp16ebs8 *__restrict scratch) {\n  aie::set_rounding(aie::rounding_mode::conv_even);\n  aie::block_vector_output_buffer_stream<bfp16ebs8,64> packed(scratch);\n  AIE_LOOP_RANGE(2, 2)\n  for (unsigned part')
    swap('''        bfloat16 *__restrict dst = scratch + sub * (8 * GQD_S);
        const unsigned ib8 = ib0 / 2;
        aie::store_v(dst + (ib8 + 0) * GQD_COLA * (8 * GQD_S), z.first);
        aie::store_v(dst + (ib8 + 1) * GQD_COLA * (8 * GQD_S), z.second);''',
         '        gqp_probe_emit(z.first,packed);\n        gqp_probe_emit(z.second,packed);')
    swap('    bfloat16 *__restrict scratch) {\n  event0();',
         '    bfp16ebs8 *__restrict scratch) {\n  event0();')
    mode = 'high-only' if high_only else 'high/residual'
    return f'// Generated compensated64 adaptation: packed BFP {mode} output.\n' + source


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--n', type=int, default=8192)
    ap.add_argument('--k', type=int, default=2048)
    turbo = ap.add_mutually_exclusive_group()
    turbo.add_argument('--turbo-high-ab', action='store_true', help='compact A plus strided activation-high DMA gather; same external ABI')
    turbo.add_argument('--turbo-high-only', action='store_true', help='HH-only with compact local weight highs; same external high/low ABI')
    turbo.add_argument('--turbo-hh', action='store_true', help='experimental HH-only MAC; ignores low limbs, retains high/low ABI')
    ap.add_argument('--weight-depth', type=int, choices=(1, 2), default=1)
    ap.add_argument('--compare-cdo', type=Path, help='existing build directory; reject a different hardware image')
    args = ap.parse_args()
    compact_a = args.turbo_high_only or args.turbo_high_ab
    if args.n <= 0 or args.n % 512 or args.k <= 0 or args.k % 256:
        ap.error('N must be a positive multiple of512 and K a positive multiple of256')
    if args.turbo_high_ab and args.k // 64 > 64 and (args.k % 4096 or args.k // 4096 > 64):
        ap.error('high-B gather requires K <=4096 or K multiple4096 up to262144')
    if args.weight_depth == 2 and not args.turbo_high_ab:
        ap.error('weight depth2 only fits the compact high-AB mode')
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / 'target'):
        ap.error('generated probes must stay under the ignored target directory')
    out.mkdir(parents=True, exist_ok=False)
    generated = out / 'source'
    generated.mkdir()
    parent = HERE.parent / 'gemm_q4_dequant.h'
    watched = [parent, HERE.parent / 'gemm_bfp_compensated_emit.h', HERE / 'design.py', HERE / 'mm.cc', Path(__file__)]
    if compact_a:
        watched += [HERE / 'high_only_emit.h', HERE / 'mm_high_only.cc']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in watched}
    header = generated / 'dequant.h'
    header.write_text(fused_header(parent.read_text(), compact_a))
    for ky in range(4):
        (generated / f'entry{ky}.cc').write_text(
            f'#include "{header}"\nextern "C" void gqd_entry_k{ky}(const uint8_t *a,uint8_t *b,bfp16ebs8 *c) {{gqd_dequant_ky(a,{ky},b,c);}}\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GQP_', 'COMPENSATED64_'))}
    env.update(COMPENSATED64_SOURCE_DIR=str(generated), COMPENSATED64_N=str(args.n), COMPENSATED64_K=str(args.k), OFLM_KEEP_FAILED='1')
    env['COMPENSATED64_WEIGHT_DEPTH'] = str(args.weight_depth)
    if compact_a:
        env['COMPENSATED64_HIGH_ONLY'] = '1'
    if args.turbo_high_ab:
        env['COMPENSATED64_HIGH_B'] = '1'
    if args.turbo_hh:
        env['COMPENSATED64_TURBO_HH'] = '1'
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
                    precision='bfp16-hh' if (args.turbo_hh or compact_a) else 'bfp16-hilo',
                    producer='high-only' if compact_a else 'high/residual',
                    local_weight_panel_bytes=4608 if compact_a else 9216,
                    weight_fifo_depth=args.weight_depth,
                    local_activation_panel_bytes=4608 if args.turbo_high_ab else 9216,
                    activation_dma='gather-high-only' if args.turbo_high_ab else 'high/low',
                    arithmetic='HH only; low limbs ignored' if (args.turbo_hh or compact_a) else 'HH + HL + LH; LL omitted',
                    source_sha256=hashes, cdo_sha256=cdo, comparison=comparison,
                    artifacts={name: digest(out / 'build' / name) for name in ['final.xclbin', 'insts.bin', 'insts.elf']},
                    hardware_validation='not performed by builder')
    (out / 'build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if comparison is not None and not comparison['identical']:
        raise RuntimeError('CDO payload differs; inspect build.json, do not mix streams')
    print(json.dumps(dict(out=str(out), cdo_identical=None if comparison is None else comparison['identical']), indent=2))


if __name__ == '__main__':
    main()
