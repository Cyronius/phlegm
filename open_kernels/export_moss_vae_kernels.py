"""Export the MOSS Image SFT decoder's native NPU kernel streams.

This is a decoder-only export for numerical validation. Its set markers and
instruction streams are consumed with the packed VAE weights and decoder plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from open_kernels import export_dit_kernels as export
from open_kernels.vae_decoder import MOSS_BLOCK_OUT, stream_specs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", default="512", help="comma-separated square image sizes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    resolutions = [int(r) for r in args.resolutions.split(",")]
    if not resolutions or any(r not in (512, 1024) for r in resolutions):
        raise ValueError("MOSS VAE currently supports 512 and 1024 pixel squares")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    sets = stream_specs(resolutions, MOSS_BLOCK_OUT)
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
                raise ValueError(f"unsupported MOSS VAE convolution {name}: {exc}") from exc
    for name, spec in sets.get("fa", {}).items():
        why = fa_check(spec["L"], spec["heads"], spec["kv_heads"])
        if why:
            raise ValueError(f"{name}: {why}")
    for name, spec in sets.get("gemm", {}).items():
        why = gemm_check(spec["M"], spec["K"], spec["N"])
        if why:
            raise ValueError(f"{name}: {why}")
    args.out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for kset in ("conv1", "conv", "vew", "rgba", "gemm", "fa"):
        specs = sets.get(kset, {})
        if not specs:
            continue
        out = args.out / kset
        out.mkdir(exist_ok=True)
        marker = out / "set.json"
        marker.unlink(missing_ok=True)
        design_set = "vew" if kset == "rgba" else kset
        jobs = {name: export.stream_job(design_set, spec, out) for name, spec in specs.items()}
        print(f"{kset}: building {len(jobs)} MOSS VAE streams", flush=True)
        dirs = export.build_many(jobs, False, args.jobs)
        export.assemble(out, dirs, marker, {
            "family": "OpenMOSS-Team/MOSS-Image-SFT",
            "kernel_set": kset,
            "resolutions": resolutions,
            "profile": list(MOSS_BLOCK_OUT),
            "streams": specs,
        })
        entries = [out / "final.xclbin", marker]
        entries += [out / f"insts_{name}.bin" for name in specs]
        for path in entries:
            files[str(path.relative_to(args.out))] = {"bytes": path.stat().st_size,
                                                     "sha256": digest(path)}
    (args.out / "manifest.json").write_text(json.dumps({
        "family": "OpenMOSS-Team/MOSS-Image-SFT",
        "profile": list(MOSS_BLOCK_OUT),
        "resolutions": resolutions,
        "sets": {k: len(v) for k, v in sets.items()},
        "files": files,
        "complete": True,
    }, indent=2, sort_keys=True) + "\n")
    print(f"exported {sum(map(len, sets.values()))} MOSS VAE streams -> {args.out}")


if __name__ == "__main__":
    main()
