#!/usr/bin/env python3
"""Check the pristine engine import against a local Git tree (no checkout edits)."""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATHS = ("src/open_qwen36", "src/include/nlohmann", "LICENSE_OPEN_RUNTIME.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--commit", default="38ae3d2")
    args = parser.parse_args()
    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(args.upstream), *cmd])
    expected = set(git("ls-tree", "-r", "--name-only", args.commit, "--", *PATHS).decode().splitlines())
    actual = {p.relative_to(ROOT).as_posix() for name in PATHS
              for p in ([ROOT / name] if (ROOT / name).is_file() else (ROOT / name).rglob("*"))
              if p.is_file()}
    failures = [f"file set differs: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"] if expected != actual else []
    for name in sorted(expected & actual):
        if hashlib.sha256((ROOT / name).read_bytes()).digest() != hashlib.sha256(git("show", f"{args.commit}:{name}")).digest():
            failures.append(f"sha256 mismatch: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(expected)} imported files byte-identical to {args.commit}")


if __name__ == "__main__":
    main()
