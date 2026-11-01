"""Export MOSS latent-entry and velocity-exit native NPU streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from open_kernels import export_dit_kernels as export
from open_kernels.moss_image_edges import specs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    sets = specs(args.resolution)
    from open_kernels.designs.dit_ew.dit_ew import check_spec
    from open_kernels.designs.dit_gemm.dit_gemm import check_shape as gemm_check
    for name, spec in sets["bias"].items():
        if why := check_spec(spec):
            raise ValueError(f"EW {name}: {why}")
    for name, spec in sets["gemm"].items():
        if why := gemm_check(spec["M"], spec["K"], spec["N"]):
            raise ValueError(f"GEMM {name}: {why}")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for kset, streams in sets.items():
        out = root / kset
        out.mkdir(exist_ok=True)
        marker = out / "set.json"
        marker.unlink(missing_ok=True)
        design_set = "ew" if kset == "bias" else kset
        jobs = {name: export.stream_job(design_set, spec, out)
                for name, spec in streams.items()}
        print(f"{kset}: building {len(jobs)} MOSS edge streams", flush=True)
        dirs = export.build_many(jobs, False, args.jobs)
        export.assemble(out, dirs, marker, {
            "family": "OpenMOSS-Team/MOSS-Image-SFT",
            "resolution": args.resolution, "streams": streams})
        for path in [out / "final.xclbin", marker,
                     *(out / f"insts_{name}.bin" for name in streams)]:
            files[str(path.relative_to(root))] = {"bytes": path.stat().st_size,
                                                    "sha256": digest(path)}
    (root / "manifest.json").write_text(json.dumps({
        "family": "OpenMOSS-Team/MOSS-Image-SFT",
        "scope": "native image entry and final velocity projection",
        "resolution": args.resolution, "files": files, "complete": True,
    }, indent=2, sort_keys=True) + "\n")
    print(f"exported {sum(map(len, sets.values()))} image edge streams -> {root}")


if __name__ == "__main__":
    main()
