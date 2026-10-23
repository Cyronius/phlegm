#!/usr/bin/env python3
"""Interleave two open-engine builds on the SAME token ids and context capacity.

Measures engine prefill/decode, not HTTP TTFT. Saves commands, hashes, stdout,
stderr and every sample; warmups are excluded from medians. Quality is a separate
compare-open-logits.py run. Does not download, serve or modify any model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path

PREFILL = re.compile(r"prefill (\d+) tokens: ([\d.]+) ms")
DECODE = re.compile(r"decode (\d+) tokens: ([\d.]+) ms/token \(([\d.]+) tok/s\)")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def run(cli, kernels, model, ids_file, n, max_tokens, ctx, pmode, env, prefix, timeout):
    command = [str(cli.resolve()), "--model", str(model.resolve()), "--kernels", str(kernels.resolve()),
               "--ids-file", str(ids_file.resolve()), "--max-tokens", str(max_tokens),
               "--max-ctx", str(ctx), "--pmode", pmode, "--gemm-block"]
    p = subprocess.run(command, env=env, capture_output=True, timeout=timeout)
    prefix.with_suffix(".stdout").write_bytes(p.stdout)
    prefix.with_suffix(".stderr").write_bytes(p.stderr)
    text = p.stderr.decode("utf-8", errors="replace")
    pre, dec = PREFILL.search(text), DECODE.search(text)
    tokens = re.findall(rb"^token (\d+)\r?$", p.stdout, re.M)
    if p.returncode or not pre or not dec or int(pre[1]) != n or len(tokens) != max_tokens:
        raise RuntimeError(f"invalid sample {prefix}: exit {p.returncode}, inspect its raw logs")
    if int(dec[1]) != max_tokens - 1:
        raise RuntimeError(f"wrong decode count in {prefix}")
    ms = float(pre[2])
    if ms <= 0 or float(dec[3]) <= 0 or "xrt-smi failed" in text:
        raise RuntimeError(f"invalid timing or power mode in {prefix}")
    result = {"command": command, "prompt_tokens": n, "prefill_ms": ms,
              "prefill_tps": n * 1000 / ms, "decode_tps": float(dec[3]),
              "allocation_log": [s for s in text.splitlines() if "buffer bytes:" in s]}
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-cli", required=True, type=Path)
    ap.add_argument("--candidate-cli", required=True, type=Path)
    ap.add_argument("--reference-kernels", required=True, type=Path)
    ap.add_argument("--candidate-kernels", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--ids-file", required=True, type=Path)
    ap.add_argument("--sizes", default="1024,2048,4096")
    ap.add_argument("--reference-attention", choices=("0", "1"))
    ap.add_argument("--candidate-attention", choices=("0", "1"))
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--pmode", default="performance")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    ids = [int(s) for s in re.split(r"[\s,]+", a.ids_file.read_text().strip()) if s]
    sizes = [int(s) for s in a.sizes.split(",")]
    if not sizes or min(sizes) < 1 or max(sizes) > len(ids) or max(sizes)+a.max_tokens > a.max_ctx:
        ap.error("ids-file must cover every size and max-ctx must fit prompt plus decode")
    if a.max_tokens < 2 or a.reps < 1 or a.warmups < 0:
        ap.error("need at least two output tokens, one repetition and nonnegative warmups")
    a.out.mkdir(parents=True, exist_ok=False)
    metadata = {"platform": platform.platform(), "processor": platform.processor(),
                "context_capacity": a.max_ctx, "pmode": a.pmode, "warmups": a.warmups,
                "reps": a.reps, "model_sha256": sha(a.model / "model.q4nx"),
                "ids_sha256": sha(a.ids_file), "env": {k:v for k,v in os.environ.items()
                   if k.startswith("OFLM_OPEN_") or k in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY")},
                "variants": {}}
    variants = {"reference": (a.reference_cli, a.reference_kernels), "candidate": (a.candidate_cli, a.candidate_kernels)}
    for label, (cli, kernels) in variants.items():
        metadata["variants"][label] = {"cli_sha256": sha(cli), "kernels": str(kernels.resolve()),
            "manifest_sha256": sha(kernels / "manifest.json"),
            "attention_override": getattr(a, label + "_attention"),
            "toolchain": json.loads((kernels / "toolchain.json").read_text())}
    (a.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    samples = []
    for n in sizes:
        ids_file = a.out / f"ids-{n}.txt"
        ids_file.write_text(",".join(map(str, ids[:n])))
        for r in range(-a.warmups, a.reps):
            order = list(variants) if r % 2 == 0 else list(reversed(variants))
            for label in order:
                cli, kernels = variants[label]
                env = dict(os.environ)
                override = getattr(a, label + "_attention")
                if override is not None:
                    env["OFLM_OPEN_ATTN_BLOCK"] = override
                result = run(cli, kernels, a.model, ids_file, n, a.max_tokens, a.max_ctx, a.pmode,
                             env, a.out / f"{label}-{n}-{r}", a.timeout)
                result.update(variant=label, repetition=r, warmup=r < 0)
                samples.append(result)
                print(f"{label} N={n} rep={r}: prefill {result['prefill_tps']:.1f}, decode {result['decode_tps']:.2f} tok/s", flush=True)
    summary = []
    for n in sizes:
        for label in variants:
            rows = [s for s in samples if s["variant"] == label and s["prompt_tokens"] == n and not s["warmup"]]
            summary.append({"variant": label, "prompt_tokens": n, **{
                key: {"median": statistics.median(s[key] for s in rows), "min": min(s[key] for s in rows), "max": max(s[key] for s in rows)}
                for key in ("prefill_ms", "prefill_tps", "decode_tps")}})
    (a.out / "results.json").write_text(json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
