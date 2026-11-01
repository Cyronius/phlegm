"""MOSS prompt embedding through the native Qwen3.5 NPU encoder.

This is the image runtime's text-tower seam. It runs one unpadded prompt at a
time, as the reference does for Qwen GDN, and refuses missing native assets.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .moss_text_tokens import MAX_SEQUENCE_LENGTH, token_ids


HIDDEN = 4096


def encode_prompt(
    prompt: str,
    *,
    tokenizer_dir: Path,
    model_dir: Path,
    kernel_dir: Path,
    encoder_binary: Path,
    max_length: int = 4096,
) -> np.ndarray:
    """Return float32 ``[real_tokens, 4096]`` final hidden states from NPU.

    ``model_dir`` contains the MOSS-derived headless ``model.q4nx`` and its
    ``config.json``. ``kernel_dir`` is the matching Qwen3.5 9B native kernel
    set. The binary is ``open_qwen36_cli`` with ``--encode-hidden`` support.
    """
    model_dir, kernel_dir, encoder_binary = map(Path, (model_dir, kernel_dir, encoder_binary))
    for path in (model_dir / "model.q4nx", model_dir / "config.json",
                 kernel_dir / "manifest.json", encoder_binary):
        if not path.is_file():
            raise FileNotFoundError(path)
    ids = token_ids(Path(tokenizer_dir), prompt, max_length)
    with tempfile.TemporaryDirectory(prefix="moss-text-") as directory:
        ids_file = Path(directory) / "ids.txt"
        hidden_file = Path(directory) / "hidden.f32"
        ids_file.write_text(",".join(map(str, ids)), encoding="ascii")
        subprocess.run(
            [str(encoder_binary), "--model", str(model_dir), "--kernels", str(kernel_dir),
             "--ids-file", str(ids_file), "--max-ctx", str(MAX_SEQUENCE_LENGTH),
             "--pmode", "none", "--encode-hidden", str(hidden_file)],
            check=True,
        )
        expected = len(ids) * HIDDEN * 4
        actual = hidden_file.stat().st_size
        if actual != expected:
            raise ValueError(f"native encoder wrote {actual} bytes; expected {expected}")
        return np.fromfile(hidden_file, dtype="<f4").reshape(len(ids), HIDDEN)
