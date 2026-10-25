"""Compact projection pools must keep the qualified head and attention contracts."""
from pathlib import Path

import pytest

from recipes.cache import build_key
from recipes.load import load_spec
from recipes.manifest import manifest

SPEC = Path(__file__).resolve().parents[3] / 'open_kernels/recipes/specs/llama32-1b.json'


def test_compact_pool_changes_storage_and_kernel_selection_without_changing_head(monkeypatch):
    for flag in ('DENSE_GGUF_GEMM', 'DENSE_GGUF_HEAD_Q8', 'DENSE_FLASH_PREFILL'):
        monkeypatch.setenv(flag, '1')
    monkeypatch.delenv('DENSE_GGUF_COMPACT', raising=False)
    spec = load_spec(SPEC)
    before = manifest(spec)
    key = build_key(spec)
    monkeypatch.setenv('DENSE_GGUF_COMPACT', '1')
    after = manifest(spec)
    assert build_key(spec) != key
    # The q4nx contract remains unchanged.
    for field in ('layer_types', 'layout', 'pack', 'tail'):
        assert after[field] == before[field]
    old, new = before['gguf'], after['gguf']
    assert new['layout']['chunk_bytes'] == new['pack']['chunk_bytes'] == 5120
    assert new['layout']['pool_bytes'] < old['layout']['pool_bytes']
    assert new['pack']['lm_head'] == old['pack']['lm_head']
    assert new['layout']['lmhead_pool_bytes'] == old['layout']['lmhead_pool_bytes']
    assert new['globals']['lmpool'] == old['globals']['lmpool']
    assert new['tail'] == old['tail']
    g = new['layer_types']['dense']['gemm_block']
    assert g['attn_fa'] == old['layer_types']['dense']['gemm_block']['attn_fa']
    assert g['attn_fa']['psplit']
    for step in g['program']:
        assert not step['kernel'].endswith('_f32')
        assert step['kernel'] in new['kernels']
    for op, base in zip(new['layer_types']['dense']['pack']['pool'], after['layer_types']['dense']['pack']['pool']):
        assert op['op'] == 'std_perm_gguf'
        assert (op['tensor'], op['dst'], op['nch']) == (base['tensor'], base['dst'], base['nch'])


def test_compact_mode_refuses_missing_head_prerequisite(monkeypatch):
    monkeypatch.setenv('DENSE_GGUF_COMPACT', '1')
    monkeypatch.setenv('DENSE_GGUF_GEMM', '1')
    monkeypatch.delenv('DENSE_GGUF_HEAD_Q8', raising=False)
    with pytest.raises(ValueError, match='requires DENSE_GGUF_GEMM=1 and DENSE_GGUF_HEAD_Q8=1'):
        manifest(load_spec(SPEC))
