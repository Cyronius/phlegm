"""Nemotron-H's kernel TUs (recipes/nemotronh.py GEN_KERNELS): none are generated -- every nh_*.cc is a checked-in
source whose shapes arrive as runtime arguments (K, the band law) or from the recipe at trace time (nh.py). The
exporter stages the design tree through this hook, so it exists and writes nothing."""
from __future__ import annotations

from pathlib import Path


def generate(recipe, out: Path) -> list[Path]:
    return []
