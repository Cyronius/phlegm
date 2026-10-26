"""Turbo publication must verify precision provenance and shared hardware image."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'utilities/gguf-prefill/prepare_bfp_hilo_image.py'
NAMES = ['gemm_n3072_k2048_f32', 'gemm_n2048_k2048_f32',
         'gemm_n8192_k2048_f32', 'gemm_n2048_k8192_f32']


def fixture(tmp_path, packed=False):
    base = tmp_path / 'base'
    base.mkdir()
    program = [dict(kernel=n, args=['weights', 'input', 'output'],
                    **({'fmt': 'bfp16_hilo'} if packed else {})) for n in NAMES]
    gg = dict(layout=dict(chunk_bytes=6144), globals=dict(input=1152 if packed else 1024),
              layer_types=dict(dense=dict(gemm_block=dict(t=256, qw=2048, kvw=512,
                  ff=8192, attn_fa=dict(psplit=True), program=program))))
    (base / 'manifest.json').write_text(json.dumps(dict(gguf=gg)))
    command = [sys.executable, str(SCRIPT), '--turbo', '--base', str(base),
               '--out', str(base / 'turbo')]
    builds = []
    for tag in ['qkv', 'o', 'up', 'down']:
        build = tmp_path / tag / 'build'
        cdo = build / 'final.prj/cdo_main'
        cdo.mkdir(parents=True)
        for name in ['main_aie_cdo_elfs.bin', 'main_aie_cdo_init.bin', 'main_aie_cdo_enable.bin']:
            (cdo / name).write_bytes(name.encode())
        hashes = {}
        for name in ['final.xclbin', 'insts.bin', 'insts.elf']:
            payload = (name + tag).encode()
            (build / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        n, k = dict(qkv=(3072, 2048), o=(2048, 2048), up=(8192, 2048), down=(2048, 8192))[tag]
        (build.parent / 'build.json').write_text(json.dumps(dict(n=n, k=k, tokens=256, precision='bfp16-hh',
            arithmetic='HH only; low limbs ignored', weight_chunk_bytes=6144,
            input='bfp16_hilo', output='float32[N,T]', artifacts=hashes)))
        command += ['--' + tag, str(build)]
        builds.append(build)
    return base, builds, command


def test_turbo_bundle_preserves_packed_sizes_and_base(tmp_path):
    base, builds, command = fixture(tmp_path, packed=True)
    original = (base / 'manifest.json').read_bytes()
    subprocess.run(command, check=True, capture_output=True)
    assert (base / 'manifest.json').read_bytes() == original
    marker = json.loads((base / 'turbo/turbo.json').read_text())
    assert marker['base_manifest_sha256'] == hashlib.sha256(original).hexdigest()
    assert marker['turbo_manifest_sha256'] == hashlib.sha256((base / 'turbo/manifest.json').read_bytes()).hexdigest()
    manifest = json.loads((base / 'turbo/manifest.json').read_text())
    assert manifest['gguf']['globals']['input'] == 1152
    assert not (base / 'turbo/turbo').exists()


def test_turbo_rejects_changed_payload_and_mixed_image(tmp_path):
    base, builds, command = fixture(tmp_path)
    binary = builds[1] / 'insts.bin'
    saved = binary.read_bytes()
    binary.write_bytes(b'changed')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode and 'artifact differs' in result.stderr
    assert not (base / 'turbo').exists()
    binary.write_bytes(saved)
    (builds[1] / 'final.prj/cdo_main/main_aie_cdo_elfs.bin').write_bytes(b'different image')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode and 'image differs' in result.stderr
    assert not (base / 'turbo').exists()


def test_turbo_rejects_wrong_projection_shape(tmp_path):
    base, builds, command = fixture(tmp_path)
    path = builds[0].parent / 'build.json'
    metadata = json.loads(path.read_text())
    metadata['n'] = 8192
    path.write_text(json.dumps(metadata))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode and 'shape does not match' in result.stderr
    assert not (base / 'turbo').exists()
