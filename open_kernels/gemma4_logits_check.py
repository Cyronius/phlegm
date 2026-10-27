"""Check Gemma 4 NPU q4 head output through the NPU final logit cap."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from gemma4_lm_head_check import VOCAB_BANDS, WIDTHS, fixture


def run(driver: Path, cap_image: Path, head_run: Path, out: Path, model: str) -> dict:
    head_report = json.loads((head_run / "report.json").read_text())
    if not head_report["passed"] or head_report["model"] != model or \
            head_report["vocab_logits"] != VOCAB_BANDS * 64:
        raise ValueError("requires a passing full-vocabulary NPU head check for this model")
    source = head_run / "result.bin"
    head_logits = np.fromfile(source, np.float32)
    if head_logits.size != VOCAB_BANDS * 64 or not np.isfinite(head_logits).all():
        raise ValueError("invalid head result")
    width = WIDTHS[model]
    _, activation, first = fixture(width, 20261027)
    _, _, second = fixture(width, 20261028, activation)
    reference_head = np.tile(np.concatenate((first, second)), VOCAB_BANDS // 2)
    expected = (30 * np.tanh(reference_head.astype(np.float64) / 30)).astype(np.float32)
    for name in ("final.xclbin", "insts.bin"):
        if not (cap_image / name).is_file():
            raise FileNotFoundError(cap_image / name)
    out.mkdir(parents=True, exist_ok=True)
    config = out / "run.cfg"
    config.write_text("\n".join([
        "device", f"xclbin G {cap_image / 'final.xclbin'}",
        f"kernelx k G {cap_image / 'insts.bin'}",
        f"buf logits {source.stat().st_size} {source}",
        f"buf capped {source.stat().st_size}", "run k logits capped",
        f"dump capped {out / 'capped.bin'} {source.stat().st_size}", "",
    ]))
    proc = subprocess.run((str(driver), str(config)), capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"NPU harness failed: {proc.stdout}\n{proc.stderr}")
    actual = np.fromfile(out / "capped.bin", np.float32)
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise AssertionError("invalid capped logits")
    a, e = actual.astype(np.float64), expected.astype(np.float64)
    maximum = float(np.max(np.abs(a - e)))
    cosine = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e)))
    report = {"model": model, "vocab_logits": len(actual),
              "max_absolute_error": maximum, "cosine": cosine,
              "passed": bool(maximum < 0.0002 and cosine > 0.99999999),
              "reference": "FP64 decoded q4 head followed by 30*tanh(logits/30)"}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(report)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", required=True, type=Path)
    p.add_argument("--cap-image", required=True, type=Path)
    p.add_argument("--head-run", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", choices=WIDTHS, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.driver.resolve(), a.cap_image.resolve(), a.head_run.resolve(),
                         a.out.resolve(), a.model), indent=2))


if __name__ == "__main__":
    main()
