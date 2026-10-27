"""Standalone GPT-OSS gates reject malformed/nonfinite data without NPU work."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name):
    path = ROOT / f'open_kernels/designs/gptoss_{name}/check.py'
    spec = importlib.util.spec_from_file_location(f'gptoss_{name}_checker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = load('gate')
ROUTER = load('router')


def test_gate_accepts_known_valid_values_and_retains_quality_limit():
    expected = np.linspace(-1, 1, 64, dtype=np.float32)
    assert GATE.validate(expected.copy(), expected) == (0, 0)
    with pytest.raises(AssertionError):
        GATE.validate(expected + .02, expected)


@pytest.mark.parametrize('target', ['output', 'reference'])
@pytest.mark.parametrize('bad', [np.full(64, np.nan), np.full(64, np.inf),
                                 np.zeros(1), np.zeros(63), np.zeros(65), np.zeros((8, 8))])
def test_gate_rejects_nonfinite_or_wrong_count(target, bad):
    actual, expected = np.zeros(64), np.zeros(64)
    if target == 'output':
        actual = bad
    else:
        expected = bad
    with pytest.raises(AssertionError):
        GATE.validate(actual, expected)


def valid_router():
    logits = np.arange(32, dtype=np.float64)
    ids = np.array([31, 30, 29, 28], dtype=np.int32)
    weights = np.exp(np.array([0., -1., -2., -3.]))
    weights /= weights.sum()
    return ids, weights, logits, logits.copy()


def test_router_accepts_independent_known_valid_values_and_stable_ties():
    assert ROUTER.validate(*valid_router()) == (0, 0)
    assert ROUTER.validate(np.arange(4), np.full(4, .25), np.zeros(32), np.zeros(32)) == (0, 0)


@pytest.mark.parametrize('field,bad', [
    (0, np.array([31, 30, 29])), (0, np.array([31., 30., 29., 28.])),
    (0, np.array([31, 31, 29, 28])), (0, np.array([32, 30, 29, 28])),
    (0, np.array([-1, 30, 29, 28])),
    (1, np.array([np.nan, .2, .3, .5])), (1, np.full(4, np.inf)),
    (1, np.full(3, 1/3)), (1, np.full(5, .2)),
    (1, np.array([-.01, .3, .3, .41])), (1, np.full(4, .3)),
    (2, np.full(32, np.nan)), (2, np.full(32, np.inf)), (2, np.zeros(31)),
    (3, np.full(32, np.nan)), (3, np.full(32, np.inf)), (3, np.zeros(33)),
])
def test_router_rejects_invalid_ids_counts_finiteness_and_normalization(field, bad):
    args = list(valid_router())
    args[field] = bad
    with pytest.raises(AssertionError):
        ROUTER.validate(*args)


def test_router_quality_bounds_are_unchanged():
    ids, weights, logits, expected = valid_router()
    with pytest.raises(AssertionError):
        ROUTER.validate(ids, weights, logits + .003, expected)
    changed = weights.copy()
    changed[0] += .003
    changed[1] -= .003
    with pytest.raises(AssertionError):
        ROUTER.validate(ids, changed, logits, expected)


def test_router_rejects_weights_above_one_even_within_error_tolerance():
    logits = np.zeros(32)
    logits[0] = 1000
    ids = np.arange(4)
    assert ROUTER.validate(ids, np.array([1., 0., 0., 0.]), logits, logits) == (0, 0)
    with pytest.raises(AssertionError, match=r'\[0,1\]'):
        ROUTER.validate(ids, np.array([1.001, 0., 0., 0.]), logits, logits)


@pytest.mark.parametrize('kind,size', [('gate', 126), ('gate', 129), ('gate', 130),
                                      ('router', 252), ('router', 257)])
def test_raw_output_requires_exact_byte_count_without_device(monkeypatch, tmp_path, kind, size):
    checker = GATE if kind == 'gate' else ROUTER
    if kind == 'gate':
        cases = [('invalid', np.zeros(64, np.float32), np.zeros(64, np.float32))]
        output = 'hidden.bin'
    else:
        cases = [('invalid', np.zeros(2880, np.float32), np.zeros((2880, 32), np.float32), np.zeros(32, np.float32))]
        output = 'result.bin'
    monkeypatch.setattr(checker, 'cases', lambda *args: cases)

    def fake_process(*args, **kwargs):
        (tmp_path / 'invalid' / output).write_bytes(bytes(size))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(checker.subprocess, 'run', fake_process)
    with pytest.raises(AssertionError, match='output bytes'):
        checker.run(Path('unused-driver'), Path('unused-build'), tmp_path)
