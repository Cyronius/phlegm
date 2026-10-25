"""The extra-core probe must preserve ownership, stream sharing and cache identity."""
from pathlib import Path

from recipes.attnknobs import kv_cores, kv_slices
from recipes.cache import build_key
from recipes.load import load_spec

SPEC = Path(__file__).resolve().parents[3] / 'open_kernels/recipes/specs/llama32-1b.json'


def test_llama_extra_cores_are_explicit_and_share_whole_kv_head_pairs(monkeypatch):
    spec = load_spec(SPEC)
    monkeypatch.delenv('ATTN_G4CORES', raising=False)
    key = build_key(spec)
    assert kv_cores(spec, 32, 8, 64, 4, 4) == 0
    monkeypatch.setenv('ATTN_G4CORES', '1')
    assert build_key(spec) != key
    assert kv_cores(spec, 32, 8, 64, 4, 4) == 1
    width, starts = kv_slices(32, 8, 8, share=2)
    assert width == 2 and starts == [0, 0, 2, 2, 4, 4, 6, 6]
    for core, start in enumerate(starts):
        assert start <= core < start + width
    # No silent expansion to shapes whose code or DMA budget was not checked.
    assert kv_cores(spec, 32, 8, 64, 4, 0) == 0
    assert kv_cores(spec, 32, 8, 128, 4, 4) == 0
