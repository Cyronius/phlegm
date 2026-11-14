"""Compile-only V1 selector: source types, layer boundaries and legacy identity."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from recipes import dense
from recipes.catalogue import OpRangeError
from recipes.cache import source_files
from recipes.dense_precision import PROJECTIONS, from_gguf_types, layer_formats, projection_spec
from recipes.load import HERE, load_spec
from recipes.spec import ModelSpec, SpecError


def base():
    return load_spec(HERE / 'specs/llama32-1b.json')


def selected(promoted=(1, 7)):
    types = {f'blk.{i}.{p}.weight': 'Q6_K' if i in promoted and p in ('attn_v', 'ffn_down') else 'Q4_K'
             for i in range(16) for p in PROJECTIONS}
    return replace(base(), tensor_quant=from_gguf_types(types, 16))


def generator():
    path = HERE.parent / 'designs/dense/gen_kernels.py'
    spec = importlib.util.spec_from_file_location('dense_tensor_gen', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_selection_follows_file_not_fixed_pattern():
    for promoted in ((1, 7), (0, 2, 15)):
        spec = selected(promoted)
        for i in range(16):
            got = {p for p, f in layer_formats(spec, i).items() if f == 'q8'}
            assert got == ({'attn_v', 'ffn_down'} if i in promoted else set())


def test_roundtrip_hash_and_legacy_serialization():
    old = base()
    assert 'tensor_quant' not in old.to_dict()
    assert replace(old, tensor_quant={}).spec_hash() == old.spec_hash()
    new = selected()
    assert ModelSpec.from_json(new.to_json()) == new
    assert new.spec_hash() != old.spec_hash()
    assert new.kernel_hash() != old.kernel_hash()
    assert new.quant_hash() != old.quant_hash()


@pytest.mark.parametrize('selector', [[], {'blk.16.attn_v.weight': 'q8'},
    {'blk.1.typo.weight': 'q8'}, {'blk.1.attn_v.weight': 'Q6_K'},
    {'blk.01.attn_v.weight': 'q8'}, {'blk.-1.attn_v.weight': 'q8'}])
def test_invalid_selectors_rejected(selector):
    with pytest.raises(SpecError, match='tensor_quant'):
        ModelSpec.from_dict({**base().to_dict(), 'tensor_quant': selector})


@pytest.mark.parametrize('layer', ['-1', '16', 'garbage', ''])
def test_explicit_valid_layer_required(monkeypatch, layer):
    monkeypatch.setenv('DENSE_GGUF_LAYER', layer)
    with pytest.raises(SpecError, match='DENSE_GGUF_LAYER'):
        dense.recipe(selected())


def test_missing_projection_refused(monkeypatch):
    spec = selected()
    del spec.tensor_quant['blk.1.attn_q.weight']
    monkeypatch.setenv('DENSE_GGUF_LAYER', '1')
    with pytest.raises(SpecError, match='missing projections'):
        dense.recipe(spec)


def test_unsupported_source_type_names_tensor():
    with pytest.raises(SpecError, match=r'blk.0.attn_v.weight.*IQ4_XS'):
        from_gguf_types({'blk.0.attn_v.weight': 'IQ4_XS'}, 16)


def test_v1_pool_and_bodies(monkeypatch):
    spec = selected()
    monkeypatch.setenv('DENSE_GGUF_LAYER', '1')
    r = dense.recipe(spec)
    fs = generator().files(r)
    assert {'gemv_q4_gy.cc', 'gemv_q4_gms.cc', 'gemv_q8_gy.cc'} <= fs.keys()
    assert 'gemv_q8_gms.cc' not in fs
    assert 'gemv_q4_gyms.cc' not in fs
    # q8 uses four padded half-tiles; attention Q/K/O and FFN up/gate stay q4.
    def size(p, rows, cols):
        return dense.role_bytes(projection_spec(spec, p), 'attn' if p.startswith('attn_') else 'ffn', rows, cols)
    assert r.layout.POOL_O - r.layout.POOL_V == size('attn_v', 512, 2048) == 2 * dense.q4_bytes(512, 2048)
    assert r.layout.POOL_GATE - r.layout.POOL_UP == dense.q4_bytes(8192, 2048)
    assert r.geo.CALL_BYTES == dense.recipe(base()).geo.CALL_BYTES
    assert r.geo.TAB_BYTES == dense.recipe(base()).geo.TAB_BYTES
    with pytest.raises(OpRangeError, match='compile-only'):
        dense.pack_plan(spec)
    assert dense.gemm_route(spec) is None
    assert dense.probe_env()['DENSE_GGUF_LAYER'] == '1'


def test_v0_generates_legacy_sources(monkeypatch):
    monkeypatch.setenv('DENSE_GGUF_LAYER', '0')
    assert generator().files(dense.recipe(selected())) == generator().files(dense.recipe(base()))


def test_other_q8_projection_is_not_silently_compiled(monkeypatch):
    spec = selected()
    spec.tensor_quant['blk.1.ffn_up.weight'] = 'q8'
    monkeypatch.setenv('DENSE_GGUF_LAYER', '1')
    with pytest.raises(OpRangeError, match='attn_v / ffn_down only'):
        dense.recipe(spec)


def test_q8_header_is_a_build_input_only_when_used():
    header = HERE.parent / 'designs/gemv_q4/gemv_q8.h'
    assert header not in source_files(base())
    assert header in source_files(selected())


def test_combined_head_stream_is_refused_for_probe(monkeypatch):
    monkeypatch.setenv('DENSE_GGUF_LAYER', '1')
    monkeypatch.setenv('DX_ONECTX', '1')
    with pytest.raises(OpRangeError, match='DX_ONECTX=0'):
        dense.recipe(selected())


@pytest.mark.parametrize('q8', [set(), {'attn_v'}, {'ffn_down'}, {'attn_v', 'ffn_down'}])
def test_projection_formats_are_independent(monkeypatch, q8):
    spec = selected(())
    for p in q8:
        spec.tensor_quant[f'blk.3.{p}.weight'] = 'q8'
    monkeypatch.setenv('DENSE_GGUF_LAYER', '3')
    r = dense.recipe(spec)
    assert {p for p, fmt in layer_formats(spec).items() if fmt == 'q8'} == q8
    assert ('gemv_q8_gy.cc' in generator().files(r)) == bool(q8)
