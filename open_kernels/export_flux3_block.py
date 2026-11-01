"""Export native kernel streams for a FLUX 3 checkpoint-weighted image block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from open_kernels import export_dit_kernels as export
from open_kernels.flux3_block import plan
from open_kernels.flux3_shared_block import plan as shared_plan


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def reuse_set(donor: Path, out: Path, specs: dict) -> bool:
    """Copy a compiled kernel set whose stream specs equal ``specs`` exactly.

    Saves rebuilding identical streams; the reused set is byte-identical to the
    one already validated on hardware."""
    if not (donor / "set.json").is_file():
        return False
    marker = json.loads((donor / "set.json").read_text())
    if not marker.get("complete") or marker["streams"] != json.loads(json.dumps(specs)):
        return False
    import shutil
    for name in ["final.xclbin", "toolchain.json", *(f"insts_{n}.bin" for n in specs)]:
        if (donor / name).is_file():
            shutil.copyfile(donor / name, out / name)
    (out / "set.json").write_text(json.dumps(marker | {"copied_from": str(donor)}, indent=2) + "\n")
    print(f"{out.name}: reused {len(specs)} identical streams from {donor}", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--mode", choices=("image", "image_cond", "text"), default="image")
    parser.add_argument("--stage", choices=("early", "shared"), default="early")
    parser.add_argument("--text-length", type=int, default=80)
    parser.add_argument("--references", type=int, default=0,
                        help="shared stage: same-size reference images (0 or 1)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--reuse", type=Path,
                        help="kernel directory whose identically specified sets are copied")
    args = parser.parse_args()
    p = (shared_plan(args.resolution, args.text_length, args.layer, args.references)
         if args.stage == "shared" else
         plan(args.resolution, args.layer, mode=args.mode, text_length=args.text_length))
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
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sets = {"gemm": p.streams["gemm"],
            "ew": {k: v for k, v in p.streams["ew"].items() if k != "qk"},
            "qk": {"qk": p.streams["ew"]["qk"]},
            "fa": p.streams["fa"]}
    files = {}
    for kset, specs in sets.items():
        out = root / kset
        out.mkdir(exist_ok=True)
        marker = out / "set.json"
        marker.unlink(missing_ok=True)
        if args.reuse and reuse_set(args.reuse.resolve() / kset, out, specs):
            for file in [out / "final.xclbin", marker,
                         *(out / f"insts_{name}.bin" for name in specs)]:
                files[str(file.relative_to(root))] = {
                    "bytes": file.stat().st_size, "sha256": digest(file)}
            continue
        design = "ew" if kset == "qk" else kset
        jobs = {name: export.stream_job(design, spec, out)
                for name, spec in specs.items()}
        print(f"{kset}: compiling {len(jobs)} FLUX 3 streams", flush=True)
        dirs = export.build_many(jobs, False, args.jobs)
        export.assemble(out, dirs, marker, {
            "family": "black-forest-labs/flux-3-image-klein",
            "resolution": args.resolution, "layer": args.layer,
            "stage": args.stage, "mode": args.mode if args.stage == "early" else "joint",
            "text_length": args.text_length if args.stage == "shared" or args.mode == "text" else 0,
            "streams": specs,
        })
        for file in [out / "final.xclbin", marker,
                     *(out / f"insts_{name}.bin" for name in specs)]:
            files[str(file.relative_to(root))] = {
                "bytes": file.stat().st_size, "sha256": digest(file)}
    (root / "manifest.json").write_text(json.dumps({
        "family": "black-forest-labs/flux-3-image-klein",
        "scope": "one native FLUX 3 transformer block",
        "resolution": args.resolution, "layer": args.layer,
        "stage": args.stage, "mode": args.mode if args.stage == "early" else "joint",
        "text_length": args.text_length if args.stage == "shared" or args.mode == "text" else 0,
        **({"references": args.references} if args.stage == "shared" and args.references else {}),
        "tokens": p.tokens, "valid_tokens": p.valid_tokens,
        "files": files, "complete": True,
    }, indent=2) + "\n")
    print(f"exported FLUX 3 block kernels -> {root}")


if __name__ == "__main__":
    main()
