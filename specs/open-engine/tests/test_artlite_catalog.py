"""Issue #212: small models must export complete open-kernel sets."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("filename,model", [
    ("llama32-1b", "Llama-3.2-1B-NPU2"),
    ("llama32-3b", "Llama-3.2-3B-NPU2"),
    ("qwen3-0.6b", "Qwen3-0.6B-NPU2"),
])
def test_small_models_have_complete_default_export_specs(monkeypatch, filename, model):
    monkeypatch.delenv("OPEN_KERNELS_UNVALIDATED", raising=False)
    from recipes.load import load_spec
    from recipes.manifest import manifest

    spec = load_spec(ROOT / "open_kernels/recipes/specs" / (filename + ".json"))
    assert spec.extra["model"] == model
    m = manifest(spec)
    assert len(m["layers"]) == spec.num_layers
    assert {"dx", "ln", "lm_head_q4", "dx_f32", "lm_head_q4_f32", "dx_attn"} <= m["builds"].keys()
    for kernel in m["kernels"].values():
        assert kernel["build"] in m["builds"]
    assert m["layer_types"]["dense"]["gemm_block"]["program"]
