"""Derive a MOSS kernel set for another real text length in the same NPU bucket.

Two independently exported lengths serve as a witness: their q/k and FA
instruction streams must differ *only* at the repeated RTP words containing
the text length or joint valid length. Everything else is linked from the
base set. This avoids compiling one xclbin per ordinary prompt length.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from open_kernels.moss_dit_block import plan as block_plan
from open_kernels.moss_text_refine import plan as text_plan


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_sets(kind: str, resolution: int, length: int) -> dict:
    if kind == "text":
        p = text_plan(length)
        return {
            "gemm": p.streams["gemm"],
            "ew": {k: v for k, v in p.streams["ew"].items()
                   if k in ("norm", "residual")},
            "bias": {k: v for k, v in p.streams["ew"].items()
                     if k.startswith("bias")},
            "qk": {"qk": p.streams["ew"]["qk"]},
            "situ": {"situ": p.streams["ew"]["situ"]},
            "fa": p.streams["fa"],
        }
    if kind == "block":
        p = block_plan(resolution, length)
        return {
            "gemm": p.streams["gemm"],
            "ew": {k: v for k, v in p.streams["ew"].items()
                   if k not in ("qk", "situ")},
            "qk": {"qk": p.streams["ew"]["qk"]},
            "situ": {"situ": p.streams["ew"]["situ"]},
            "fa": p.streams["fa"],
        }
    raise ValueError(f"unknown MOSS kernel kind {kind}")


def _patch(base: Path, witness: Path, before: int, after: int,
           target: int, count: int, stride: int) -> bytes:
    a = np.fromfile(base, dtype="<u4")
    b = np.fromfile(witness, dtype="<u4")
    if a.size != b.size:
        raise ValueError(f"RTP witness stream changed size: {base}")
    changed = np.flatnonzero(a != b)
    if (changed.size != count or
            not np.all(np.diff(changed) == stride) or
            not np.all(a[changed] == before) or
            not np.all(b[changed] == after)):
        raise ValueError(f"RTP witness differs outside expected length words: {base}")
    a[changed] = target
    return a.tobytes()


def derive(base: Path, witness: Path, out: Path, kind: str,
           resolution: int, length: int) -> Path:
    base, witness, out = (Path(v).resolve() for v in (base, witness, out))
    bm = json.loads((base / "manifest.json").read_text())
    wm = json.loads((witness / "manifest.json").read_text())
    old, other = bm["text_length"], wm["text_length"]
    if old == other or length in (old, other):
        raise ValueError("variant length must differ from both exported witnesses")
    if (not bm.get("complete") or not wm.get("complete") or
            bm.get("tokens") != wm.get("tokens") or
            bm.get("family") != wm.get("family") or
            bm.get("resolution") != wm.get("resolution") or
            (kind == "block" and bm.get("resolution") != resolution)):
        raise ValueError("MOSS kernel witnesses have different physical geometry")
    target = text_plan(length) if kind == "text" else block_plan(resolution, length)
    if target.tokens != bm["tokens"]:
        raise ValueError("requested text length crosses a padded NPU token bucket")
    sets = stream_sets(kind, resolution, length)
    for directory, manifest, source_length in ((base, bm, old),
                                                (witness, wm, other)):
        expected = stream_sets(kind, resolution, source_length)
        if manifest["files"].keys() != bm["files"].keys():
            raise ValueError("MOSS witness kernel file inventory differs")
        for name, specs in expected.items():
            marker = json.loads((directory / name / "set.json").read_text())
            if not marker.get("complete") or marker["streams"] != specs:
                raise ValueError(f"MOSS {name} witness stream differs from its plan")
        for name, entry in manifest["files"].items():
            path = directory / name
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError(f"MOSS witness file failed integrity check: {path}")
    for name in sets:
        a, b, c = (stream_sets(kind, resolution, n)[name]
                   for n in (old, other, length))
        if name == "qk":
            for key in a:
                if key != "qk" and (a[key] != b[key] or a[key] != c[key]):
                    raise ValueError("MOSS q/k witness physical stream changed")
            for field in a["qk"]:
                if field != "n_txt" and (a["qk"][field] != b["qk"][field]
                                         or a["qk"][field] != c["qk"][field]):
                    raise ValueError("MOSS q/k witness changed beyond text length")
        elif name == "fa":
            stream = "text" if kind == "text" else "joint"
            for field in a[stream]:
                if field != "valid_len" and (a[stream][field] != b[stream][field]
                                             or a[stream][field] != c[stream][field]):
                    raise ValueError("MOSS attention witness changed beyond valid length")
        elif a != b or a != c:
            raise ValueError(f"MOSS {name} witness changed physical stream")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    patch_names = {"qk/insts_qk.bin": (old, other, length, 16, 144),
                   f"fa/insts_{'text' if kind == 'text' else 'joint'}.bin":
                   (text_plan(old).text_length if kind == "text" else
                    block_plan(resolution, old).valid_tokens,
                    text_plan(other).text_length if kind == "text" else
                    block_plan(resolution, other).valid_tokens,
                    target.text_length if kind == "text" else target.valid_tokens,
                    32, 36)}
    files = {}
    for name in bm["files"]:
        path = out / name
        path.parent.mkdir(exist_ok=True)
        if name in patch_names:
            path.write_bytes(_patch(base / name, witness / name, *patch_names[name]))
        elif name.endswith("/set.json"):
            kset = name.split("/", 1)[0]
            marker = json.loads((base / name).read_text())
            marker["text_length"] = length
            marker["streams"] = sets[kset]
            path.write_text(json.dumps(marker, indent=2) + "\n")
        else:
            path.symlink_to(base / name)
        files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest = bm.copy()
    manifest["text_length"] = length
    manifest["files"] = files
    manifest["derived_from"] = {"base": str(base),
                                 "base_sha256": digest(base / "manifest.json"),
                                 "witness": str(witness),
                                 "witness_sha256": digest(witness / "manifest.json"),
                                 "method": "verified q/k and FA RTP length patch"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out
