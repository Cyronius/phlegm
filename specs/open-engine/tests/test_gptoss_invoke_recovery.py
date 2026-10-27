"""The GPT-OSS harness wrapper records a bounded replay of invalid output."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from open_kernels.designs.gptoss_attention import check_chain


def test_invalid_first_dump_replays_once(tmp_path: Path):
    driver = tmp_path / "driver"
    driver.touch()
    build = tmp_path / "build"
    build.mkdir()
    (build / "final.xclbin").touch()
    (build / "insts.bin").touch()
    output = tmp_path / "output"
    launches = []

    def fake_run(command, **kwargs):
        launches.append(command)
        values = np.array([np.nan, 0], np.float32) if len(launches) == 1 else \
            np.array([1, 2], np.float32)
        (output / "result.bin").write_bytes(values.tobytes())
        return SimpleNamespace(returncode=0)

    with patch.object(check_chain.subprocess, "run", side_effect=fake_run):
        actual = check_chain.invoke(driver, build, output,
                                    [("x", np.zeros(2, np.float32))],
                                    np.dtype(np.float32), 2)
    np.testing.assert_array_equal(actual, np.array([1, 2], np.float32))
    assert len(launches) == 2
    assert json.loads((output / "retries.json").read_text()) == {"replays": 1}
