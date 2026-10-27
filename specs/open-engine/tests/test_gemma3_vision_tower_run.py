"""Check the 27-layer tower chain and its alternating hidden-state files."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_kernels import gemma3_vision_tower_run as tower


class TowerScheduleTest(unittest.TestCase):
    def test_all_layers_feed_the_next_layer_and_final_projector(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "tower"
            seen = []

            def patch_run(_driver, _bundle, _image, _weights, directory):
                directory.mkdir(parents=True)
                path = directory / "result.bin"
                path.write_bytes(bytes([0]))
                return path

            def run_layer(_driver, _tower, _post, _attention, _residual,
                          directory, _weights, input_path, index):
                self.assertEqual(index, len(seen))
                value = input_path.read_bytes()[0]
                self.assertEqual(value, index)
                seen.append(index)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "result.bin").write_bytes(bytes([value + 1]))
                return {"passed": True}

            def finish_run(_driver, _layernorm, _pool, _projector,
                           hidden_path, _weights, _directory):
                self.assertEqual(hidden_path.read_bytes(), bytes([27]))
                return {"passed": True}

            with (patch.object(tower, "read_bf16", return_value=None),
                  patch.object(tower, "Q4NX", return_value=object()),
                  patch.object(tower, "patch_embedding", return_value={}),
                  patch.object(tower, "patch_run", side_effect=patch_run),
                  patch.object(tower, "run_layer", side_effect=run_layer),
                  patch.object(tower, "finish_run", side_effect=finish_run)):
                result = tower.run(*([Path(root) / "unused"] * 10), out)
            self.assertEqual(seen, list(range(27)))
            self.assertTrue(result["passed"])
            self.assertEqual(json.loads((out / "report.json").read_text())["layers"], 27)


if __name__ == "__main__":
    unittest.main()
