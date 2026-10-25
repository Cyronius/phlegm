"""Quality-gate boundaries must not turn a failed or partial run into a pass."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
loader = importlib.util.spec_from_file_location('gguf_quality_gate', ROOT / 'utilities/gguf-prefill/compare_model.py')
gate = importlib.util.module_from_spec(loader)
loader.loader.exec_module(gate)


def fixture(tmp_path):
    cpu = np.tile(np.array([2, 0], '<f4'), (100, 1))
    cpu.tofile(tmp_path / 'cpu.bin')
    for name in ('seq', 'candidate'):
        for i in range(100):
            cpu[i].tofile(tmp_path / f'{name}_p{i}.bin')
    # The penultimate position's opposite prediction improves loss, so the
    # second assertion below isolates argmax allowance rather than loss drift.
    ids = [0] * 99 + [1]
    return tmp_path / 'cpu.bin', tmp_path / 'seq', tmp_path / 'candidate', ids


def test_exact_one_percentage_point_is_allowed_but_two_is_not(tmp_path):
    args = fixture(tmp_path)
    np.array([0, 2], '<f4').tofile(tmp_path / 'candidate_p99.bin')
    assert gate.compare(*args)['passed']
    np.array([0, 2], '<f4').tofile(tmp_path / 'candidate_p98.bin')
    assert not gate.compare(*args)['passed']


def test_nonfinite_and_missing_positions_are_errors(tmp_path):
    args = fixture(tmp_path)
    np.array([np.nan, 0], '<f4').tofile(tmp_path / 'candidate_p50.bin')
    with pytest.raises(ValueError, match='bad logits'):
        gate.compare(*args)
    (tmp_path / 'candidate_p50.bin').unlink()
    with pytest.raises(ValueError, match='position coverage'):
        gate.compare(*args)
