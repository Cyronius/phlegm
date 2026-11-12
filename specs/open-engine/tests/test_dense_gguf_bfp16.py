"""DENSE_GGUF_BFP16: the compact dense GGUF view's fast (bfp16) and lean (q4) prefill routes."""
from pathlib import Path

from recipes.cache import build_key
from recipes.load import load_spec
from recipes.manifest import manifest
from recipes import dense

SPEC = Path(__file__).resolve().parents[3] / 'open_kernels/recipes/specs/minicpm5-2b.json'
FLAGS = ('DENSE_GGUF_GEMM', 'DENSE_GGUF_HEAD_Q8', 'DENSE_GGUF_COMPACT')


def test_bfp16_is_fast_and_the_q4_route_stays_as_lean(monkeypatch):
    for flag in FLAGS:
        monkeypatch.setenv(flag, '1')
    monkeypatch.delenv('DENSE_GGUF_BFP16', raising=False)
    spec = load_spec(SPEC)
    before = manifest(spec)
    key = build_key(spec)
    assert not any('bfp16' in k for k in dense.builds(spec))
    monkeypatch.setenv('DENSE_GGUF_BFP16', '1')
    after = manifest(spec)
    assert build_key(spec) != key
    # the q4nx view and its manifest version are untouched
    for field in ('layer_types', 'layout', 'pack', 'tail', 'manifest_version'):
        assert after[field] == before[field]
    old, new = before['gguf']['layer_types'], after['gguf']['layer_types']
    for name, layer in new.items():
        # lean is exactly the route the compact view ran before
        assert layer['gemm_block_variants'] == {'lean': old[name]['gemm_block']}
        fast = layer['gemm_block']
        assert fast['t'] == old[name]['gemm_block']['t'] and fast['kind'] == old[name]['gemm_block']['kind']
        assert all(s['fmt'] == 'bfp16' and s['kernel'].startswith('gemm_bfp16_') for s in fast['program'])
        assert all(s['kernel'] in after['gguf']['kernels'] for s in fast['program'])
        # no GLU epilogue: the in-tree engine reads manifest_version <= 4
        assert not any(s.get('epi') for s in fast['program'])
    assert after['gguf']['manifest_version'] == 4
    assert {k for k in dense.builds(spec) if 'bfp16' in k} == {
        'gemm_bfp16_n3072_k2048', 'gemm_bfp16_n2048_k2048', 'gemm_bfp16_n6144_k2048', 'gemm_bfp16_n2048_k6144'}
