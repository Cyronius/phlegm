"""Compile one FLUX 3 half-scale image-coordinate q/k NPU validation stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_kernels import export_dit_kernels as export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if spec.get("op") != "qk_flux3":
        raise ValueError("expected a FLUX 3 q/k stream spec")
    from open_kernels.designs.dit_ew.dit_ew import check_spec
    if why := check_spec(spec):
        raise ValueError(why)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dirs = export.build_many({"qk": export.stream_job("ew", spec, out)}, False, 1)
    export.assemble(out, dirs, out / "set.json", {
        "family": "black-forest-labs/flux-3-image-klein",
        "scope": "half-scale image-coordinate q/k probe",
        "streams": {"qk": spec},
    })


if __name__ == "__main__":
    main()
