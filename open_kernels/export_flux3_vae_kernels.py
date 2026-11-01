"""Export native NPU decoder kernels for the pinned FLUX 3 autoencoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from open_kernels import export_dit_kernels as export
from open_kernels.vae_decoder import BLOCK_OUT, stream_specs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="512")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    resolutions = [int(value) for value in args.resolutions.split(",")]
    if not resolutions or any(value not in (512, 1024) for value in resolutions):
        raise ValueError("FLUX 3 VAE supports 512 and 1024 pixel squares")
    sets = stream_specs(resolutions, BLOCK_OUT)
    sets["rgba"] = {name: spec for name, spec in sets["vew"].items()
                    if spec["op"] == "rgba"}
    sets["vew"] = {name: spec for name, spec in sets["vew"].items()
                   if spec["op"] != "rgba"}
    from open_kernels.designs.dit_fa.dit_fa import check_shape as fa_check
    from open_kernels.designs.dit_gemm.dit_gemm import check_shape as gemm_check
    from open_kernels.designs.dit_conv.dit_conv import resolve_spec, tiles
    for kset in ("conv", "conv1"):
        for name, spec in sets.get(kset, {}).items():
            try:
                tiles(resolve_spec(spec, 9 if kset == "conv" else 1))
            except AssertionError as exc:
                raise ValueError(f"FLUX 3 VAE convolution {name}: {exc}") from exc
    for name, spec in sets.get("fa", {}).items():
        if why := fa_check(spec["L"], spec["heads"], spec["kv_heads"]):
            raise ValueError(f"FA {name}: {why}")
    for name, spec in sets.get("gemm", {}).items():
        if why := gemm_check(spec["M"], spec["K"], spec["N"]):
            raise ValueError(f"GEMM {name}: {why}")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for kset in ("conv1", "conv", "vew", "rgba", "gemm", "fa"):
        specs = sets.get(kset, {})
        if not specs:
            continue
        out = root / kset
        out.mkdir(exist_ok=True)
        marker = out / "set.json"
        marker.unlink(missing_ok=True)
        design = "vew" if kset == "rgba" else kset
        jobs = {name: export.stream_job(design, spec, out)
                for name, spec in specs.items()}
        print(f"{kset}: building {len(jobs)} FLUX 3 VAE streams", flush=True)
        dirs = export.build_many(jobs, False, args.jobs)
        export.assemble(out, dirs, marker, {
            "family": "black-forest-labs/flux-3-image-klein",
            "kernel_set": kset, "resolutions": resolutions,
            "profile": list(BLOCK_OUT), "streams": specs,
        })
        for file in [out / "final.xclbin", marker,
                     *(out / f"insts_{name}.bin" for name in specs)]:
            files[str(file.relative_to(root))] = {
                "bytes": file.stat().st_size, "sha256": digest(file)}
    (root / "manifest.json").write_text(json.dumps({
        "family": "black-forest-labs/flux-3-image-klein",
        "profile": list(BLOCK_OUT), "resolutions": resolutions,
        "sets": {key: len(value) for key, value in sets.items()},
        "files": files, "complete": True,
    }, indent=2, sort_keys=True) + "\n")
    print(f"exported {sum(map(len, sets.values()))} FLUX 3 VAE streams -> {root}")


if __name__ == "__main__":
    main()
