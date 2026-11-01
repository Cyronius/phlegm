"""Compile FLUX 3 flow-map timestep and modulation kernels for the NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from open_kernels import export_dit_kernels as export
from open_kernels.flux3_conditioning import specs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    sets = specs()
    from open_kernels.designs.dit_ew.dit_ew import check_spec
    from open_kernels.designs.dit_gemm.dit_gemm import check_shape
    for name, spec in sets["ew"].items():
        if why := check_spec(spec):
            raise ValueError(f"FLUX 3 conditioning {name}: {why}")
    for name, spec in sets["gemm"].items():
        if why := check_shape(spec["M"], spec["K"], spec["N"]):
            raise ValueError(f"FLUX 3 conditioning {name}: {why}")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for kset, streams in sets.items():
        out = root / kset
        out.mkdir(exist_ok=True)
        marker = out / "set.json"
        marker.unlink(missing_ok=True)
        jobs = {name: export.stream_job(kset, spec, out)
                for name, spec in streams.items()}
        print(f"{kset}: building {len(jobs)} FLUX 3 conditioning streams", flush=True)
        dirs = export.build_many(jobs, False, args.jobs)
        export.assemble(out, dirs, marker, {
            "family": "black-forest-labs/flux-3-image-klein",
            "streams": streams,
        })
        for file in [out / "final.xclbin", marker,
                     *(out / f"insts_{name}.bin" for name in streams)]:
            files[str(file.relative_to(root))] = {
                "bytes": file.stat().st_size, "sha256": digest(file)}
    (root / "manifest.json").write_text(json.dumps({
        "family": "black-forest-labs/flux-3-image-klein",
        "scope": "native flow-map timestep and mode modulation",
        "files": files, "complete": True,
    }, indent=2, sort_keys=True) + "\n")
    print(f"exported FLUX 3 conditioning -> {root}")


if __name__ == "__main__":
    main()
