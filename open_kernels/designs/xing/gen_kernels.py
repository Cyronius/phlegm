"""Xing's kernel TUs (recipes/xing.py GEN_KERNELS): none are generated -- every xg_*.cc is a checked-in source whose
shapes arrive as runtime arguments (K, the band law) or from the recipe at trace time (xg.py; XG_W_OFF as a flag).
The exporter stages the design tree through this hook, so it exists and writes nothing."""
from __future__ import annotations

from pathlib import Path


def generate(recipe, out: Path) -> list[Path]:
    return []
