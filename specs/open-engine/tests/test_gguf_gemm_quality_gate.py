"""Independent source and BF16 operator gates keep fixed profile bounds."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
loader = importlib.util.spec_from_file_location('gguf_gemm_gate', ROOT / 'utilities/gguf-prefill/check_gemm.py')
gate = importlib.util.module_from_spec(loader)
loader.loader.exec_module(gate)


def test_strict_default_thresholds_and_exact_output():
    reference = np.eye(2)
    result = gate.evaluate_outputs(reference, reference, reference)
    assert result['profile'] == 'strict'
    assert result['thresholds'] == dict(source_rel_fro_max=.005, bf16_operand_rel_fro_max=.0001)
    assert result['passed']
    assert result['rel_fro'] == result['operand_rel_fro'] == 0


@pytest.mark.parametrize('profile,source_bound,operand_bound', [('strict', .005, .0001), ('turbo', .02, .02)])
@pytest.mark.parametrize('which', ['source', 'operand'])
def test_independent_reference_boundaries(profile, source_bound, operand_bound, which):
    # Orthogonal perturbation gives exactly epsilon relative Frobenius error.
    reference = np.array([1., 0.])
    bound = source_bound if which == 'source' else operand_bound
    for multiplier, passed in [(1., True), (1.001, False)]:
        got = np.array([1., bound * multiplier])
        source, operand = (reference, got) if which == 'source' else (got, reference)
        result = gate.evaluate_outputs(got, source, operand, profile)
        assert result['passed'] == passed
        assert result['profile'] == profile
        assert result['thresholds'] == gate.PROFILES[profile]


def test_turbo_accepts_preregistered_precision_range_only():
    reference = np.array([1., 0.])
    got = np.array([1., .0093])
    assert not gate.evaluate_outputs(got, reference, reference)['passed']
    assert gate.evaluate_outputs(got, reference, reference, 'turbo')['passed']
    # A shape-correct timing-only/zero-output ablation still fails numerically.
    assert not gate.evaluate_outputs(np.zeros(2), reference, reference, 'turbo')['passed']


@pytest.mark.parametrize('profile', ['strict', 'turbo'])
@pytest.mark.parametrize('bad', [np.array([np.nan, 0]), np.array([np.inf, 0]), np.ones(3), np.array([])])
def test_invalid_outputs_rejected(profile, bad):
    with pytest.raises(ValueError, match='incomplete or nonfinite'):
        gate.evaluate_outputs(bad, np.ones(2), np.ones(2), profile)


def test_invalid_reference_and_profile_rejected():
    with pytest.raises(ValueError, match='reference norm'):
        gate.evaluate_outputs(np.ones(2), np.zeros(2), np.ones(2))
    with pytest.raises(ValueError, match='unknown quality profile'):
        gate.evaluate_outputs(np.ones(2), np.ones(2), np.ones(2), 'timing-only')
