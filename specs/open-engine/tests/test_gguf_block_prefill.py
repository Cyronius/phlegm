"""Experimental GGUF prefill must name distinct, correctly formatted images."""
from pathlib import Path
import pytest

from recipes.cache import build_key
from recipes.load import load_spec
from recipes.manifest import manifest

SPEC = Path(__file__).resolve().parents[3] / 'open_kernels/recipes/specs/llama32-1b.json'


def test_flash_probe_replaces_score_pv_only_in_the_opt_in_gguf_route(monkeypatch):
    spec = load_spec(SPEC)
    monkeypatch.setenv('DENSE_GGUF_GEMM', '1')
    monkeypatch.delenv('DENSE_FLASH_PREFILL', raising=False)
    before = manifest(spec)
    key = build_key(spec)
    monkeypatch.setenv('DENSE_FLASH_PREFILL', '1')
    after = manifest(spec)
    assert build_key(spec) != key
    assert after['layer_types'] == before['layer_types']
    route = after['gguf']['layer_types']['dense']['gemm_block']
    assert 'attn_block' not in route
    assert route['attn_fa']['qsplit']
    assert route['attn_fa']['psplit']
    # The Q4 GEMM returns host rows, not the BFP16 c-tiled prep input.
    assert 'prep' not in route['attn_fa']
    for name in route['attn_fa']['kernels'].values():
        build = after['builds'][name]
        assert build['env']['AF_HEADS'] == '32'
        assert build['env']['AF_KV_HEADS'] == '8'
        assert build['env']['AF_PSPLIT'] == '1'
    key = build_key(spec)
    monkeypatch.setenv('AF_LMAX', '2048')
    wider = manifest(spec)
    assert build_key(spec) != key
    fa = wider['gguf']['layer_types']['dense']['gemm_block']['attn_fa']
    assert fa['l_max'] == 2048 and fa['kernels']['2048'] == 'af2048'
    assert wider['builds']['af2048']['env']['AF_MAX_ROWS'] == '2048'
    assert wider['builds']['af256']['build_dir'] != after['builds']['af256']['build_dir']
    monkeypatch.setenv('AF_LMAX', '4352')
    with pytest.raises(ValueError, match='up to 4096'):
        manifest(spec)
    monkeypatch.setenv('AF_LONG_STREAM', '1')
    extended = manifest(spec)
    assert extended['gguf']['layer_types']['dense']['gemm_block']['attn_fa']['l_max'] == 4352
    assert extended['builds']['af4352']['env']['AF_LONG_STREAM'] == '1'
    assert extended['builds']['af4352']['env']['AF_MAX_ROWS'] == '4352'
    monkeypatch.setenv('AF_LMAX', '8448')
    with pytest.raises(ValueError, match='up to 8192'):
        manifest(spec)


def test_gguf_block_is_opt_in_and_has_a_distinct_build_identity(monkeypatch):
    spec = load_spec(SPEC)
    monkeypatch.delenv('DENSE_GGUF_GEMM', raising=False)
    before = manifest(spec)
    key = build_key(spec)
    assert 'gemm_block' not in before['gguf']['layer_types']['dense']
    monkeypatch.setenv('DENSE_GGUF_GEMM', '1')
    after = manifest(spec)
    assert build_key(spec) != key
    # An experimental export does not change the q4nx program or its packing law.
    assert before['layer_types'] == after['layer_types']
    assert before['pack'] == after['pack']
    assert after['gguf']['layout']['chunk_bytes'] == 6144
    route = after['gguf']['layer_types']['dense']['gemm_block']
    assert route['t'] == 256 and route['attn_block']['prep'] == 'rope'
    for step in route['program']:
        name = step['kernel']
        assert name.endswith('_f32')
        build = after['builds'][after['gguf']['kernels'][name]['build']]
        base = after['builds'][name.removesuffix('_f32')]
        assert build['env']['GQP_SCALES_F32'] == '1'
        assert build['build_dir'] != base['build_dir']
        assert 'GQP_SCALES_F32' not in base['env']
    for kernel in after['gguf']['kernels'].values():
        assert kernel['build'] in after['builds']


def test_q8_head_is_explicit_and_does_not_change_layer_weights(monkeypatch):
    spec = load_spec(SPEC)
    monkeypatch.delenv('DENSE_GGUF_HEAD_Q8', raising=False)
    before = manifest(spec)
    key = build_key(spec)
    monkeypatch.setenv('DENSE_GGUF_HEAD_Q8', '1')
    after = manifest(spec)
    assert build_key(spec) != key
    assert after['layer_types'] == before['layer_types']
    assert after['gguf']['layer_types'] == before['gguf']['layer_types']
    section = after['gguf']
    head = section['pack']['lm_head']
    assert head['ops'] == [{'op': 'lmhead_q8', 'tensor': 'lm_head.weight',
                           'dst': 0, 'chunk_bytes': 8704, 'in_dim': spec.hidden}]
    assert head['pool_bytes'] == section['globals']['lmpool'] == section['layout']['lmhead_pool_bytes']
    kernel = section['kernels']['lm']
    build = after['builds'][kernel['build']]
    assert build['design'] == 'lm_head_q8/lm_head_q8.py'
    assert build['env']['LMHEAD_SCALES_F32'] == '0'
