"""Patch Qwen Image 2.1 attention RTP words for a prompt's token count.

The three probe streams differ from their 15-token baselines only at RTP
immediates. Comparing complete streams avoids guessing transaction offsets.
The NPU still executes the original compiled kernels and all model operations.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np


BASE_TEXT = 15
PROBE_TEXT = 16
PATCHES = {"qk": (15, 16, lambda n: n),
           "fa_text": (15, 16, lambda n: n),
           "fa_image": (1039, 1040, lambda n: 1024 + n)}


def _patched(base: Path, probe: Path, before: int, after: int, value: int,
             destination: Path) -> list[int]:
    source = np.fromfile(base / "insts.bin", dtype="<u4")
    reference = np.fromfile(probe / "insts.bin", dtype="<u4")
    if source.size != reference.size:
        raise ValueError(f"Qwen RTP probe size differs: {base} vs {probe}")
    indices = np.flatnonzero(source != reference)
    if not len(indices) or np.any(source[indices] != before) or np.any(reference[indices] != after):
        raise ValueError(f"Qwen RTP probe contains changes beyond {before} -> {after}: {probe}")
    source[indices] = value
    destination.mkdir(parents=True, exist_ok=False)
    try:
        os.link(base / "final.xclbin", destination / "final.xclbin")
    except OSError:
        shutil.copy2(base / "final.xclbin", destination / "final.xclbin")
    source.tofile(destination / "insts.bin")
    return indices.tolist()


def prepare(manifest: Path, probes: Path, text_tokens: int, out: Path) -> Path:
    if not 1 <= text_tokens <= 64:
        raise ValueError("Qwen prompt kernel patch needs 1..64 text tokens")
    named = {key: Path(value).resolve() for key, value in json.loads(manifest.read_text()).items()}
    probe_paths = {key: Path(value).resolve() for key, value in json.loads(probes.read_text()).items()}
    if set(named) != {"norm", "res", "qk", "qkv", "attn_out", "mlp_in", "mlp_out",
                      "fa_text", "fa_image"} or set(probe_paths) != set(PATCHES):
        raise ValueError("Qwen block manifest or RTP probe set is incomplete")
    out.mkdir(parents=True, exist_ok=False)
    changes = {}
    for name, (before, after, value) in PATCHES.items():
        if not (named[name] / "final.xclbin").is_file() or not (probe_paths[name] / "insts.bin").is_file():
            raise FileNotFoundError(f"missing Qwen {name} base kernel or RTP probe")
        dest = out / name
        changes[name] = _patched(named[name], probe_paths[name], before, after,
                                 value(text_tokens), dest)
        named[name] = dest
    result = out / "block-kernels.json"
    result.write_text(json.dumps({k: str(v) for k, v in named.items()}, indent=2) + "\n")
    (out / "patch.json").write_text(json.dumps({"text_tokens": text_tokens,
        "base_text_tokens": BASE_TEXT, "probe_text_tokens": PROBE_TEXT,
        "rtp_word_indices": changes}, indent=2) + "\n")
    return result
