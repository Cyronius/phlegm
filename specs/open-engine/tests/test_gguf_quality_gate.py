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


def test_profile_defaults_and_cpu_forward_kl(tmp_path):
    args = fixture(tmp_path)
    result = gate.compare(*args)
    assert result['profile'] == 'strict'
    assert result['thresholds'] == gate.PROFILES['strict']
    for route in ('sequential', 'candidate'):
        for statistic in ('median', 'mean', 'max'):
            assert result[route][f'cpu_forward_kl_{statistic}'] == 0
    # All positions contribute, including the final row with no next-token loss.
    np.array([0, 2], '<f4').tofile(tmp_path / 'candidate_p99.bin')
    result = gate.compare(*args)
    expected = 2 * (np.exp(2) - 1) / (np.exp(2) + 1)
    assert result['candidate']['cpu_forward_kl_max'] == pytest.approx(expected)
    assert result['candidate']['cpu_forward_kl_mean'] == pytest.approx(expected / 100)
    assert result['candidate']['cpu_forward_kl_median'] == 0
    np.array([1, 0], '<f4').tofile(tmp_path / 'candidate_p99.bin')
    result = gate.compare(*args)
    p = np.exp([2., 0.]) / np.exp([2., 0.]).sum()
    q = np.exp([1., 0.]) / np.exp([1., 0.]).sum()
    expected = float(p @ np.log(p / q))
    assert result['candidate']['cpu_forward_kl_max'] == pytest.approx(expected)


def test_turbo_exact_five_percentage_points_allowed_but_six_not(tmp_path):
    cpu, seq, candidate, ids = fixture(tmp_path)
    # Each changed row improves next-token loss: isolate integer argmax bound.
    ids = [0] * 94 + [1] * 6
    for i in range(95, 100):
        np.array([0, 2], '<f4').tofile(tmp_path / f'candidate_p{i}.bin')
    assert gate.compare(cpu, seq, candidate, ids, 'turbo')['passed']
    assert not gate.compare(cpu, seq, candidate, ids)['passed']
    np.array([0, 2], '<f4').tofile(tmp_path / 'candidate_p94.bin')
    assert not gate.compare(cpu, seq, candidate, ids, 'turbo')['passed']


@pytest.mark.parametrize('offset,strict,turbo', [(.08, False, True), (.1, False, False)])
def test_cosine_profiles(tmp_path, offset, strict, turbo):
    cpu, seq, candidate, _ = fixture(tmp_path)
    for i in range(100):
        np.array([2, offset], '<f4').tofile(tmp_path / f'candidate_p{i}.bin')
    assert gate.compare(cpu, seq, candidate, [1] * 100)['passed'] == strict
    assert gate.compare(cpu, seq, candidate, [1] * 100, 'turbo')['passed'] == turbo


@pytest.mark.parametrize('logit,strict,turbo', [(1.999, True, True), (1.992, False, True), (1.988, False, False)])
def test_loss_profiles(tmp_path, logit, strict, turbo):
    cpu, seq, candidate, _ = fixture(tmp_path)
    for i in range(100):
        np.array([logit, 0], '<f4').tofile(tmp_path / f'candidate_p{i}.bin')
    assert gate.compare(cpu, seq, candidate, [0] * 100)['passed'] == strict
    assert gate.compare(cpu, seq, candidate, [0] * 100, 'turbo')['passed'] == turbo


@pytest.mark.parametrize('profile', ['strict', 'turbo'])
def test_invalid_values_rejected_in_both_profiles(tmp_path, profile):
    args = fixture(tmp_path)
    with pytest.raises(ValueError, match='vocabulary'):
        gate.compare(*args[:3], [-1] * 100, profile)
    np.zeros(2, '<f4').tofile(tmp_path / 'candidate_p0.bin')
    with pytest.raises(ValueError, match='undefined cosine'):
        gate.compare(*args, profile)
    np.array([np.inf, 0], '<f4').tofile(tmp_path / 'candidate_p0.bin')
    with pytest.raises(ValueError, match='bad logits'):
        gate.compare(*args, profile)


def test_unknown_profile_is_error(tmp_path):
    with pytest.raises(ValueError, match='unknown quality profile'):
        gate.compare(*fixture(tmp_path), profile='adaptive')
