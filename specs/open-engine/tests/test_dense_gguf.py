"""P5 S1: family-neutral carriage and fail-closed K-quant derivation."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import pytest
from recipes import dense
from recipes.catalogue import OpRangeError
from recipes.load import HERE, load_spec
from recipes.manifest import manifest
from recipes.spec import ModelSpec, SpecError, quant_map_from_gguf_types


@pytest.mark.parametrize('family', ['llama3', 'qwen3', 'gemma3', 'granite', 'phi3', 'qwen2', 'hunyuan'])
def test_q8_head_is_not_llama_specific(monkeypatch, family):
    monkeypatch.setenv('DENSE_GGUF_HEAD_Q8', '1')
    s = replace(load_spec(HERE / 'specs/llama32-1b.json'), family=family, quant='q4_1_f32')
    assert dense.gguf_q8_head(s)


@pytest.mark.parametrize('name', ['llama32-1b', 'qwen3-0.6b', 'qwen3-4b', 'qwen3-8b'])
def test_compact_dense_view_preserves_head_and_pool_contract(monkeypatch, name):
    for key in ('DENSE_GGUF_GEMM', 'DENSE_GGUF_HEAD_Q8', 'DENSE_GGUF_COMPACT'):
        monkeypatch.setenv(key, '1')
    m = manifest(load_spec(HERE / f'specs/{name}.json'))
    g = m['gguf']
    assert g['pack']['chunk_bytes'] == 5120
    assert g['pack']['lm_head']['ops'][0]['op'] == 'lmhead_q8'
    for lt in g['layer_types'].values():
        assert {op['op'] for op in lt['pack']['pool']} == {'std_perm_gguf'}
    assert all(k['build'] in m['builds'] for k in g['kernels'].values())


@pytest.mark.parametrize('typ,expected', [('Q4_K', {}), ('Q6_K', {'attn': 'q8'}), ('Q5_K', {'attn': 'q8'})])
def test_kquant_role_map_is_explicit(typ, expected):
    assert quant_map_from_gguf_types('llama3', {'blk.0.attn_v.weight': typ}) == expected


def test_unknown_and_mixed_types_are_not_silently_q4():
    with pytest.raises(SpecError, match='unsupported'):
        quant_map_from_gguf_types('llama3', {'blk.0.attn_v.weight': 'IQ2_XXS'})
    with pytest.raises(SpecError, match='one format'):
        quant_map_from_gguf_types('llama3', {'blk.0.attn_v.weight': 'Q6_K', 'blk.1.attn_v.weight': 'Q4_K'})


def test_metadata_can_derive_actual_tensor_selector():
    md = {'general.architecture': 'llama', 'llama.embedding_length': 2048,
          'llama.block_count': 16, 'llama.feed_forward_length': 8192,
          'llama.attention.head_count': 32, 'llama.attention.head_count_kv': 8,
          'llama.attention.layer_norm_rms_epsilon': 1e-5, 'llama.rope.freq_base': 500000.,
          'tokenizer.ggml.tokens': ['x'] * 128256}
    from recipes.dense_precision import PROJECTIONS
    types = {f'blk.{l}.{p}.weight': 'Q4_K' for l in range(16) for p in PROJECTIONS}
    types['blk.0.attn_v.weight'] = 'Q6_K'
    s = ModelSpec.from_gguf_metadata(md, tensor_types=types)
    assert s.tensor_quant['blk.0.attn_v.weight'] == 'q8'
    assert s.tensor_quant['blk.1.attn_v.weight'] == 'q4_1'
    with pytest.raises(SpecError, match='no dense projections'):
        ModelSpec.from_gguf_metadata(md, tensor_types={})
    with pytest.raises(SpecError, match='missing projections'):
        ModelSpec.from_gguf_metadata(md, tensor_types={'blk.0.attn_v.weight': 'Q6_K'})
    with pytest.raises(OpRangeError, match='variant dispatch'):
        dense.pack_plan(s)


def test_reference_rope_orders_agree():
    import numpy as np
    path = HERE.parent / 'model/dense_gguf_reference.py'
    sp = importlib.util.spec_from_file_location('dense_ref_test', path)
    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
    x = np.arange(2*3*8).reshape(2,3,8).astype(float)
    split = np.concatenate((x[...,0::2], x[...,1::2]), axis=-1)
    a = mod.rope_pairs(x, np.arange(2), 10000, 8)
    b = mod.rope_pairs(split, np.arange(2), 10000, 8, split=True)
    np.testing.assert_array_equal(b, np.concatenate((a[...,0::2], a[...,1::2]), axis=-1))


@pytest.mark.parametrize('typ,arch,head', [('Q4_K', 'llama', False), ('Q4_K', 'qwen3', False),
                                         ('Q6_K', 'llama', False), ('Q8_0', 'qwen3', False),
                                         ('Q8_0', 'llama', True), ('Q6_K', 'llama', True)])
def test_native_packer_against_independent_gguf_py(tmp_path, typ, arch, head):
    import os
    import numpy as np
    binary = os.environ.get('PHLEGM_DENSE_PACK_TEST')
    if not binary:
        pytest.skip('set PHLEGM_DENSE_PACK_TEST to the patched no-XRT gguf_pack_test')
    dep = os.environ.get('GGUF_PY')
    if dep: sys.path.insert(0, dep)
    gguf = pytest.importorskip('gguf')
    sys.path.insert(0, str(HERE.parent / 'model'))
    from dense_gguf_check import check_tensor
    from qwen35moe_gguf_check import Gguf
    rng = np.random.default_rng(123)
    rows, cols = 128, 512
    size = {'Q4_K':144, 'Q6_K':210, 'Q8_0':34}[typ]
    block = 32 if typ == 'Q8_0' else 256
    raw = rng.integers(0,256,(rows*cols//block,size), dtype=np.uint8)
    scale = np.resize(np.array([0, 1, 0x0400, 0x2211, 0x3113], '<u2'), len(raw)).view(np.uint8).reshape(-1,2)
    if typ == 'Q6_K': raw[:,208:210] = scale
    else: raw[:,:2] = scale
    if typ == 'Q4_K':
        raw[:,2:4] = scale[::-1]
        raw[:,16:] = rng.integers(2,14,(len(raw),128),dtype=np.uint8) | (rng.integers(2,14,(len(raw),128),dtype=np.uint8)<<4)
    path = tmp_path / 'synthetic.gguf'
    name = 'output.weight' if head else 'blk.0.attn_q.weight'
    w = gguf.GGUFWriter(str(path), arch)
    w.add_head_count(4); w.add_head_count_kv(4)
    w.add_tensor(name, raw.reshape(rows,-1), raw_dtype=getattr(gguf.GGMLQuantizationType,typ))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    g = Gguf(str(path), dep); g.path = path
    result = check_tensor(g, name, Path(binary), tmp_path, head=head)
    assert result['bit_exact_carriage'] and result['bit_exact_source']
