"""Build all four T256 streams and reject a mixed hardware image."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SHAPES = {'qkv': (3072, 2048), 'o': (2048, 2048),
          'up': (8192, 2048), 'down': (2048, 8192)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--weight-depth', type=int, choices=(1, 2), default=1)
    ap.add_argument('--out', required=True, type=Path)
    turbo = ap.add_mutually_exclusive_group()
    turbo.add_argument('--turbo-hh', action='store_true')
    turbo.add_argument('--turbo-high-ab', action='store_true')
    turbo.add_argument('--turbo-high-only', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    # Each stream builder also enforces that output stays under ignored target/.
    root = next(p for p in HERE.parents if (p / 'PLAN.md').exists())
    if not out.is_relative_to(root / 'target'):
        ap.error('family output must stay under ignored target/')
    out.mkdir(parents=True, exist_ok=False)
    family = {}
    for role, (n, k) in SHAPES.items():
        command = [sys.executable, str(HERE / 'build.py'), '--out', str(out / role),
                   '--n', str(n), '--k', str(k), '--weight-depth', str(args.weight_depth)]
        if args.turbo_high_ab:
            command.append('--turbo-high-ab')
        if args.turbo_high_only:
            command.append('--turbo-high-only')
        if args.turbo_hh:
            command.append('--turbo-hh')
        if family:
            command += ['--compare-cdo', str(out / 'qkv/build')]
        subprocess.run(command, check=True)
        family[role] = json.loads((out / role / 'build.json').read_text())
    (out / 'family.json').write_text(json.dumps(family, indent=2) + '\n')


if __name__ == '__main__':
    main()
