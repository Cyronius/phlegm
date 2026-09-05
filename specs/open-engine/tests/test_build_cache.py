# Traces: OPEN-BUILD-CACHE (canonical spec: specs/open-engine/spec.md)
"""The build key covers the recipe sources, the kernel sources the recipe's
designs include, the spec and the quant format -- and nothing informational."""
from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

from recipes.cache import ROOT, build_key, source_files
from recipes.load import default_spec


def copy_tree(dst: Path) -> Path:
    for f in source_files(default_spec()):
        rel = f.relative_to(ROOT)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, dst / rel)
    return dst


def test_key_is_stable_and_covers_the_sources(tmp_path):
    spec = default_spec()
    root = copy_tree(tmp_path / "ok")
    k1 = build_key(spec, root)
    assert k1 == build_key(spec, root) == build_key(spec)
    assert k1.startswith("sha256:") and len(k1) == 7 + 64
    files = [f.relative_to(root).as_posix() for f in source_files(spec, root)]
    for must in ("recipes/qwen36moe.py", "designs/layer_x/lx.py", "designs/layer_x/xcommon.py",
                 "designs/attn/attn.h", "designs/gemv_q4/gemv_q4.h", "designs/layer_x/gen_kernels.py",
                 "designs/lm_head_q8/lm_head_q8.py", "include/vecmath.h"):
        assert must in files, must


def test_a_kernel_source_edit_changes_the_key(tmp_path):
    spec = default_spec()
    root = copy_tree(tmp_path / "edit")
    before = build_key(spec, root)
    p = root / "designs" / "attn" / "attn.h"
    p.write_text(p.read_text(encoding="utf-8") + "\n// touched\n", encoding="utf-8")
    assert build_key(spec, root) != before


def test_a_recipe_edit_changes_the_key(tmp_path):
    spec = default_spec()
    root = copy_tree(tmp_path / "recipe")
    before = build_key(spec, root)
    p = root / "recipes" / "qwen36moe.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")
    assert build_key(spec, root) != before


def test_spec_and_quant_change_the_key_but_extra_does_not():
    spec = default_spec()
    k = build_key(spec)
    assert build_key(dataclasses.replace(spec, rope_theta=1e6)) != k
    assert build_key(dataclasses.replace(spec, quant="q4_k")) != k
    assert build_key(dataclasses.replace(spec, extra={"model": "another-name"})) == k
