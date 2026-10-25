"""Assemble an explicitly experimental Llama1B packed-input image.

Requires fresh streams of the same hardware image; verifies all three CDO
payloads match before sharing a context. Does not change the base bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    for tag in ['qkv', 'o', 'up', 'down']:
        ap.add_argument(f'--{tag}', required=True, type=Path)
    args = ap.parse_args()
    m = json.loads((args.base / 'manifest.json').read_text())
    gg = m['gguf']
    if gg['layout']['chunk_bytes'] != 6144 or set(gg['layer_types']) != {'dense'}:
        ap.error('requires the f32-scale pure-Q4 dense GGUF bundle')
    g = gg['layer_types']['dense']['gemm_block']
    if (g['t'], g['qw'], g['kvw'], g['ff']) != (256, 2048, 512, 8192) or not g['attn_fa'].get('psplit'):
        ap.error('requires the validated Llama1B paired-probability flash shape')
    mapping = dict(zip(['qkv', 'o', 'up', 'down'],
                       ['gemm_n3072_k2048_f32', 'gemm_n2048_k2048_f32',
                        'gemm_n8192_k2048_f32', 'gemm_n2048_k8192_f32']))
    cdo = None
    provenance = dict(kind='experimental-bfp16-hilo', base_manifest_sha256=digest(args.base / 'manifest.json'),
                      full_model_quality='pending', artifacts={})
    for tag, name in mapping.items():
        build = getattr(args, tag)
        here = {f: digest(build / 'final.prj/cdo_main' / f)
                for f in ['main_aie_cdo_elfs.bin', 'main_aie_cdo_init.bin', 'main_aie_cdo_enable.bin']}
        if cdo is not None and here != cdo:
            raise RuntimeError(f'{tag}: image differs; cannot share the hardware context')
        cdo = here
        provenance['artifacts'][name] = {f: digest(build / f) for f in ['final.xclbin', 'insts.bin', 'insts.elf']}
    for step in g['program']:
        if step['kernel'] not in mapping.values() or step.get('fmt') or step.get('split') or step.get('n_pad'):
            raise RuntimeError('unexpected projection step; refusing to reinterpret its input')
        step['fmt'] = 'bfp16_hilo'
    names = {step['args'][1] for step in g['program']}
    for name in names:
        old = gg['globals'][name]
        if not isinstance(old, int) or old % 8:
            raise RuntimeError('unexpected input buffer representation')
        gg['globals'][name] = old * 9 // 8
    # A separate directory preserves the baseline and all unchanged images.
    shutil.copytree(args.base, args.out)
    for tag, name in mapping.items():
        dest = args.out / name
        dest.mkdir(exist_ok=True)
        for f in ['final.xclbin', 'insts.bin', 'insts.elf']:
            shutil.copy2(getattr(args, tag) / f, dest / f)
    provenance['shared_cdo_sha256'] = cdo
    (args.out / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    (args.out / 'bfp-hilo-probe.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(out=str(args.out), globals={n: gg['globals'][n] for n in names}), indent=2))


if __name__ == '__main__':
    main()
