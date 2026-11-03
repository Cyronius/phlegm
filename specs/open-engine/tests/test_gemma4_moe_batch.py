"""mb4 (open_kernels/designs/gemma4_moe_batch): the geometry and host layouts the kernel and the runtime share.
The kernel itself is validated on hardware by make_test.py / compare.py (OPEN-FAMILY-GEMMA4-MOE)."""
import importlib
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parents[3] / "open_kernels" / "designs" / "gemma4_moe_batch"


def _layout():
    sys.path.insert(0, str(D))
    try:
        return importlib.import_module("layout")
    finally:
        sys.path.remove(str(D))


def test_y_layout_is_a_permutation_inverse():
    lay = _layout()
    S, rows = 2, lay.HID
    dev = np.arange(S * rows * lay.NT, dtype=np.float32)
    y = lay.y_from_dev(dev, rows)
    assert y.shape == (S, lay.NT, rows)
    assert sorted(y.reshape(-1).tolist()) == dev.tolist()
    # device element (rbg, half, r) holds band 4 rbg + r's rows 32 half .. 32 half + 31 (1024 B = 256 floats)
    rbg, half, r = 3, 1, 2
    start = ((rbg * 2 + half) * 4 + r) * 256
    band = 4 * rbg + r
    assert np.isin(dev[start:start + 256], y[0, :, band * 64 + 32 * half:band * 64 + 32 * half + 32]).all()


def test_geometry_matches_the_gemma_moe_pool():
    src = (D / "mb4.py").read_text()
    # per expert gate | up = 2 x 12 bands of 101376 B; one expert's down is the same 1216512 B
    assert 2 * (768 // 64) * (2 * (2816 // 256) * 4608) == 2433024
    assert (2816 // 64) * (2 * (768 // 256) * 4608) == 1216512
    assert "CHUNK = 4608" in src


def test_recipe_emits_the_expert_ladder(monkeypatch):
    sys.path.insert(0, str(D.parents[2]))
    try:
        from recipes import gemma4 as G4
        from recipes.spec import ModelSpec
        from test_gemma4 import HF_26B
    finally:
        sys.path.pop(0)
    monkeypatch.setenv("OPEN_KERNELS_UNVALIDATED", "1")
    spec = ModelSpec.from_hf_config(HF_26B)
    mb = G4.moe_batch(spec)
    assert list(mb["moe_batch"]["kernels"]) == ["128", "64", "32", "16", "8"]
    b = mb["builds"]["mb_s128"]
    M = G4.moe_layout(spec)
    assert b["design"] == "gemma4_moe_batch/mb4.py"
    assert b["env"]["MB_HID"] == "2816" and b["env"]["MB_FF"] == "768" and b["env"]["MB_EXPERTS"] == "128"
    assert int(b["env"]["MB_POOL_DOWN"]) == M.POOL_DOWN == 128 * 2433024
    assert M.UP_BYTES == 1216512 and M.EXPERT_UG == 2433024
    assert mb["globals"]["mb_y"] == 128 * 2816 * 8 * 4
    # on by default: the streams join the decode set's builds; G4_MOE_BATCH=0 leaves the decode set alone
    assert "mb_s128" in G4.builds(spec)
    monkeypatch.setenv("G4_MOE_BATCH", "0")
    assert "mb_s128" not in G4.builds(spec)
    monkeypatch.delenv("G4_MOE_BATCH")
    assert G4.programs(spec)["kernels"]["mb_s64"]["patch"] == "moebatch"
