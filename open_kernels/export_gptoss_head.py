"""Pack the GPT-OSS-20B q4 LM head for the 3072-wide NPU image."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))
from q4nx import Q4NX  # noqa: E402
from recipes.gptoss_source import GptOssSource, HEAD_BANDS  # noqa: E402

BAND_SHAPE = (24, 5120)


def export(source: GptOssSource, out: Path, bands: int = HEAD_BANDS) -> dict:
    """Stream the head one 64-logit band at a time into an atomic directory."""
    if not isinstance(bands, int) or bands < 1 or bands > HEAD_BANDS:
        raise ValueError(f"head bands must be in [1,{HEAD_BANDS}]")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    try:
        digest = hashlib.sha256()
        with (temp / "head.bin").open("wb") as f:
            for band in range(bands):
                chunk = source.head_band(band)
                if chunk.shape != BAND_SHAPE or chunk.dtype != np.uint8:
                    raise ValueError(f"head band {band} has wrong kernel shape")
                data = chunk.tobytes()
                f.write(data)
                digest.update(data)
        report = {"format": "gptoss-20b-q4-lm-head-v1", "bands": bands,
                  "logits": bands * 64, "hidden": 2880, "padded_hidden": 3072,
                  "complete": bands == HEAD_BANDS,
                  "head": {"path": "head.bin", "bytes": bands * int(np.prod(BAND_SHAPE)),
                           "sha256": digest.hexdigest()},
                  "scope": "q4 LM-head NPU input; no model runtime"}
        (temp / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temp, out)
        return report
    except BaseException:
        shutil.rmtree(temp)
        raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path, help="model.q4nx path")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    reader = Q4NX(a.model)
    try:
        report = export(GptOssSource(reader), a.out)
        print(json.dumps({"out": str(a.out), "logits": report["logits"],
                          "bytes": report["head"]["bytes"],
                          "sha256": report["head"]["sha256"]}, indent=2))
    finally:
        reader.mm.close()
        reader.f.close()


if __name__ == "__main__":
    main()
