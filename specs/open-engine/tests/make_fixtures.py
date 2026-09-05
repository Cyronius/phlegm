"""Write fixtures/manifest_qwen36.json: the 27B manifest with a fixed build key
(the real one hashes the sources, which would churn the fixture on every edit).

    python specs/open-engine/tests/make_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "open_kernels"))
from recipes.load import default_spec  # noqa: E402
from recipes.manifest import manifest  # noqa: E402

FIXTURE = HERE / "fixtures" / "manifest_qwen36.json"


def fixture_manifest() -> dict:
    return manifest(default_spec(), key="sha256:fixture")


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture_manifest(), indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {FIXTURE}")
