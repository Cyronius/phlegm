"""The fidelity gate must fail on missing positions, bad data and real regressions."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("compare_open_logits", ROOT / "utilities/compare-open-logits.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def dump(prefix, pos, values):
    np.array(values, dtype="<f4").tofile(str(prefix) + f"_p{pos}.bin")


def test_missing_position_fails_instead_of_scoring_the_intersection(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    dump(a, 0, [1, 2, 3]); dump(a, 1, [1, 2, 3]); dump(b, 0, [1, 2, 3])
    with pytest.raises(ValueError, match="coverage"):
        gate.compare(a, b)


def test_equal_and_wrong_logits_have_different_gate_results(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    dump(a, 0, [1, 2, 3]); dump(b, 0, [1, 2, 3])
    assert gate.compare(a, b, exact=True)["passed"]
    dump(b, 0, [1, 3, 2])
    assert not gate.compare(a, b)["passed"]
    dump(b, 0, [1, 2, float("nan")])
    with pytest.raises(ValueError, match="nonfinite"):
        gate.compare(a, b)


@pytest.mark.parametrize("returncode,prompt,outputs,power_ok", [(1,2,2,True), (0,3,2,True), (0,2,1,True), (0,2,2,False)])
def test_benchmark_refuses_failed_or_unmatched_measurements(tmp_path, monkeypatch, returncode, prompt, outputs, power_ok):
    import subprocess
    spec = importlib.util.spec_from_file_location("benchmark_open_cli", ROOT / "utilities/benchmark-open-cli.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    stderr = f"prefill {prompt} tokens: 10 ms\ndecode 1 tokens: 2 ms/token (500 tok/s)\n"
    if not power_ok:
        stderr += "NPU power mode: performance (xrt-smi failed; mode is whatever it was)"
    result = subprocess.CompletedProcess([], returncode, b"token 1\n" * outputs, stderr.encode())
    monkeypatch.setattr(bench.subprocess, "run", lambda *a, **kw: result)
    with pytest.raises(RuntimeError):
        bench.run(tmp_path / "cli", tmp_path, tmp_path, tmp_path / "ids", 2, 2, 256,
                  "performance", {}, tmp_path / "bad-sample", 1)


def test_exact_gate_distinguishes_signed_zero(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    dump(a, 0, [0.0, 1.0]); dump(b, 0, [-0.0, 1.0])
    assert not gate.compare(a, b, exact=True)["passed"]
