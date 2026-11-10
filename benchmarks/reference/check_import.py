#!/usr/bin/env python3
"""Check the in-tree engine against the OFLM import base plus the recorded changes.

Every file must be byte-identical to the base commit unless a change in
engine-changes.json lists it with its current sha256.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PATHS = ("src/open_qwen36", "src/include/nlohmann", "LICENSE_OPEN_RUNTIME.md")
CHANGES = Path(__file__).with_name("engine-changes.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--commit", default=None, help="base commit (default: engine-changes.json base)")
    args = parser.parse_args()
    record = json.loads(CHANGES.read_text())
    args.commit = args.commit or record["base"]
    changed = {}
    for change in record["changes"]:
        changed.update(change["files"])
    def git(*cmd):
        return subprocess.check_output(["git", "-C", str(args.upstream), *cmd])
    expected = set(git("ls-tree", "-r", "--name-only", args.commit, "--", *PATHS).decode().splitlines())
    actual = {p.relative_to(ROOT).as_posix() for name in PATHS
              for p in ([ROOT / name] if (ROOT / name).is_file() else (ROOT / name).rglob("*"))
              if p.is_file()}
    expected |= set(changed)
    failures = [f"file set differs: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"] if expected != actual else []
    for name in sorted(expected & actual):
        have = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        want = changed.get(name) or hashlib.sha256(git("show", f"{args.commit}:{name}")).hexdigest()
        if have != want:
            failures.append(f"sha256 mismatch: {name}" + (" (recorded change)" if name in changed else ""))
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(expected) - len(changed)} engine files byte-identical to {args.commit[:12]}, "
          f"{len(changed)} match recorded changes")


if __name__ == "__main__":
    main()
