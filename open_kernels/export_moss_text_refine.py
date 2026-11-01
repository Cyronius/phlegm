"""Export the native MOSS Image text projection and refinement streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from open_kernels import export_dit_kernels as export
from open_kernels.moss_text_refine import plan


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-length", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    p = plan(args.text_length)
    from open_kernels.designs.dit_ew.dit_ew import check_spec
    from open_kernels.designs.dit_fa.dit_fa import check_shape as fa_check
    from open_kernels.designs.dit_gemm.dit_gemm import check_shape as gemm_check
    for name, spec in p.streams["ew"].items():
        if why := check_spec(spec):
            raise ValueError(f"EW {name}: {why}")
    for name, spec in p.streams["gemm"].items():
        if why := gemm_check(spec["M"], spec["K"], spec["N"]):
            raise ValueError(f"GEMM {name}: {why}")
    for name, spec in p.streams["fa"].items():
        if why := fa_check(spec["L"], spec["heads"], spec["kv_heads"]):
            raise ValueError(f"FA {name}: {why}")
    sets = {"gemm": p.streams["gemm"],
            "ew": {k: v for k, v in p.streams["ew"].items()
                   if k in ("norm", "residual")},
            "bias": {k: v for k, v in p.streams["ew"].items()
                     if k.startswith("bias")},
            "qk": {"qk": p.streams["ew"]["qk"]},
            "situ": {"situ": p.streams["ew"]["situ"]},
            "fa": p.streams["fa"]}
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for kset, specs in sets.items():
        out = root / kset
        out.mkdir(exist_ok=True)
        marker = out / "set.json"
        marker.unlink(missing_ok=True)
        design_set = "ew" if kset in ("bias", "qk", "situ") else kset
        jobs = {name: export.stream_job(design_set, spec, out) for name, spec in specs.items()}
        print(f"{kset}: building {len(jobs)} MOSS text streams", flush=True)
        dirs = export.build_many(jobs, False, args.jobs)
        export.assemble(out, dirs, marker, {
            "family": "OpenMOSS-Team/MOSS-Image-SFT",
            "text_length": args.text_length,
            "streams": specs,
        })
        for path in [out / "final.xclbin", marker,
                     *(out / f"insts_{name}.bin" for name in specs)]:
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size,
                                                    "sha256": digest(path)}
    (root / "manifest.json").write_text(json.dumps({
        "family": "OpenMOSS-Team/MOSS-Image-SFT",
        "scope": "native text projection and two refinement blocks",
        "text_length": args.text_length,
        "tokens": p.tokens,
        "files": files,
        "complete": True,
    }, indent=2, sort_keys=True) + "\n")
    print(f"exported {sum(map(len, sets.values()))} MOSS text streams -> {root}")


if __name__ == "__main__":
    main()
