"""Projected conditioning: checked artifacts and the per-step native input boundary."""
import hashlib
import json
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ml_dtypes import bfloat16

import flux2_native_generate as fg
import klein_conditioning as kc
import klein_pipeline as kp
import flux2_conditioning_replay as replay


def fixture_tensors(dtype="BF16"):
    values = np.broadcast_to(np.arange(kc.WIDTH, dtype=np.float32) / 128, (kc.POSITIONS, kc.WIDTH)).copy()
    # Include signed zero in the payload: BF16 must pass through bit for bit.
    values[0, 0] = -0.0
    values = values.astype({"BF16": bfloat16, "F16": np.float16, "F32": np.float32}[dtype])
    positions = np.zeros((kc.POSITIONS, 4), np.int64)
    positions[:, 3] = np.arange(kc.POSITIONS)
    ids = np.full(kc.POSITIONS, kc.PAD_ID, np.int64)
    ids[:3] = [2, 3, 4]
    mask = np.zeros(kc.POSITIONS, np.int64)
    mask[:3] = 1
    return {"projected_conditioning": values, "text_ids": positions, "input_ids": ids,
            "attention_mask": mask}


def save_artifact(path, tensors, protocol=kc.PROTOCOL):
    # Small independent container fixture, using the documented safetensors header.
    dtypes = {np.dtype(bfloat16): "BF16", np.dtype("float16"): "F16", np.dtype("float32"): "F32",
              np.dtype("int64"): "I64", np.dtype("int32"): "I32"}
    header, chunks, offset = {"__metadata__": {"protocol": protocol}}, [], 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {"dtype": dtypes[array.dtype], "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks))


@pytest.mark.parametrize("dtype", ["BF16", "F16", "F32"])
def test_artifact_rounding_preserves_all_rows_and_positions(tmp_path, dtype):
    tensors = fixture_tensors(dtype)
    path = tmp_path / "input.safetensors"
    save_artifact(path, tensors)
    result = kc.load_projected_conditioning(path)
    want = tensors["projected_conditioning"].astype(bfloat16).view(np.uint16)
    assert np.array_equal(result.bits, want)
    assert np.array_equal(result.bits[-1], want[-1])  # padding is not discarded
    assert result.bits[0, 0] == 0x8000
    assert result.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.source_dtype == dtype
    result.require_prompt(tensors["input_ids"], 3)
    with pytest.raises(ValueError, match="--prompt"):
        result.require_prompt(tensors["input_ids"], 4)


@pytest.mark.parametrize("kind", ["position", "mask", "padding", "negative-id", "missing", "shape", "id-dtype", "nan", "overflow", "protocol"])
def test_invalid_artifacts_rejected(tmp_path, kind):
    tensors = fixture_tensors("F32")
    if kind == "position":
        tensors["text_ids"][3, 3] = 2
    elif kind == "mask":
        tensors["attention_mask"][:3] = [1, 0, 1]
    elif kind == "padding":
        tensors["input_ids"][-1] = 5
    elif kind == "negative-id":
        tensors["input_ids"][0] = -1
    elif kind == "missing":
        tensors.pop("input_ids")
    elif kind == "shape":
        tensors["projected_conditioning"] = tensors["projected_conditioning"][:-1]
    elif kind == "id-dtype":
        tensors["input_ids"] = tensors["input_ids"].astype(np.int32)
    elif kind == "nan":
        tensors["projected_conditioning"][0, 0] = np.nan
    elif kind == "overflow":
        tensors["projected_conditioning"][0, 0] = np.finfo(np.float32).max
    path = tmp_path / "input.safetensors"
    save_artifact(path, tensors, "other" if kind == "protocol" else kc.PROTOCOL)
    with pytest.raises(ValueError):
        kc.load_projected_conditioning(path)


def test_corrupt_and_oversized_container_rejected(tmp_path):
    path = tmp_path / "input.safetensors"
    path.write_bytes(b"corrupt")
    with pytest.raises(Exception):
        kc.load_projected_conditioning(path)
    with path.open("wb") as writer:
        writer.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="16 MiB"):
        kc.load_projected_conditioning(path)


def test_run_restores_text_before_every_step_and_drains_pending_runs(monkeypatch):
    monkeypatch.setitem(sys.modules, "npu_host", SimpleNamespace(COMPLETED="done"))
    expected = np.full((kc.POSITIONS, kc.WIDTH), 7, np.uint16)
    state = {"text": None, "pending": 0, "uploads": 0, "starts": [], "completed": []}

    class InputBuffer:
        def write(self, bits):
            assert state["pending"] == 0
            state["text"] = bits.copy()
            state["uploads"] += 1

    class Stream:
        def __init__(self, name, mutate=False):
            self.name, self.mutate = name, mutate

        def start(self):
            state["starts"].append(self.name)
            state["pending"] += 1
            def wait():
                state["pending"] -= 1
                if self.mutate:
                    assert np.array_equal(state["text"], expected)
                    state["text"][:] = 999  # the preceding step must not leak into the next
                state["completed"].append(self.name)
                return "done"
            return SimpleNamespace(wait=wait)

    runner = fg.Runner.__new__(fg.Runner)
    runner.projected, runner._conditioning_bits = True, expected.copy()
    runner.bufs = {"X": InputBuffer()}
    runner.ops = []
    for step in range(3):
        phase = f"step{step}"
        runner.ops.extend([
            ("gemm", Stream(f"image{step}"), [], phase, {"what": "x_emb"}),
            ("gemm", None, [], phase, {"what": "ctx_emb"}),
            ("gemm", Stream(f"denoise{step}", True), [], phase, {"what": "denoise"}),
        ])
    assert runner.dispatch_count == 6
    runner.run()
    assert state["uploads"] == 3
    assert state["pending"] == 0
    assert len(state["completed"]) == 6
    runner.run()  # a second request/run also starts from the original conditioning
    assert state["uploads"] == 6


def test_failed_prompt_replacement_cannot_reuse_previous_conditioning(monkeypatch):
    monkeypatch.setattr(kp, "token_ids", lambda tok, prompt: (np.array([1]), 1))
    runner = fg.Runner.__new__(fg.Runner)
    runner.projected, runner._conditioning_bits, runner.tok = True, np.ones(1), None
    def reject(*args):
        raise ValueError("mismatched prompt")
    with pytest.raises(ValueError):
        runner.set_projected_conditioning(SimpleNamespace(require_prompt=reject), "wrong")
    assert runner._conditioning_bits is None
    with pytest.raises(ValueError, match="has not been set"):
        runner.run()


def test_projected_packing_skips_teacher_and_original_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(kp, "dit_weight_specs", lambda: {"keep": (128, 128, None), "ctx_emb": (128, 128, None)})
    def forbidden():
        raise AssertionError("text weights must not be inspected")
    monkeypatch.setattr(kp, "te_weight_specs", forbidden)
    monkeypatch.setattr(fg, "fingerprint", lambda model, subs, code: "dit-only" if subs == ("transformer",) else forbidden())
    (tmp_path / "keep.bin").write_bytes(bytes(128 * 128 * 9 // 8))
    packed = fg.ensure_packed(tmp_path, tmp_path, 1, text_encoder=False)
    assert set(packed) == {"keep"}


def test_projected_checkpoint_does_not_require_teacher_config(tmp_path):
    (tmp_path / "transformer").mkdir()
    config = {"num_layers": kp.N_DBL, "num_single_layers": kp.N_SGL, "num_attention_heads": kp.HEADS,
              "attention_head_dim": kp.HD, "joint_attention_dim": 3 * kp.TE_HID, "in_channels": kp.LAT_CH}
    (tmp_path / "transformer/config.json").write_text(json.dumps(config))
    (tmp_path / "model_index.json").write_text(json.dumps({"is_distilled": True}))
    assert fg.check_checkpoint(tmp_path, projected=True)["is_distilled"]
    with pytest.raises(FileNotFoundError):
        fg.check_checkpoint(tmp_path)
    config["num_layers"] += 1
    (tmp_path / "transformer/config.json").write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="geometry"):
        fg.check_checkpoint(tmp_path, projected=True)


def test_projected_parameters_preserve_dit_slots_and_zero_unused_text_slots():
    get_dit = lambda name: np.ones(kp.HD, np.float32)
    get_te = lambda name: np.ones(kp.HD if "self_attn" in name else kp.TE_HID, np.float32)
    normal = kp.fill_params(get_dit, get_te)
    projected = kp.fill_params(get_dit)
    assert normal.shape == projected.shape
    slots, _ = kp.param_table()
    for name, (offset, size) in slots.items():
        a, b = normal[offset // 2:(offset + size) // 2], projected[offset // 2:(offset + size) // 2]
        if name.startswith("te"):
            assert not b.view(np.uint16).any()
            assert a.view(np.uint16).any()
        else:
            assert np.array_equal(a.view(np.uint16), b.view(np.uint16))


def test_projected_constructor_never_opens_teacher_or_text_streams(tmp_path, monkeypatch):
    streams, caches = [], []

    class Buffer:
        def write(self, *args):
            pass

        def zero(self):
            pass

    class KernelSet:
        def stream(self, name):
            assert name not in {"te_qkv", "te_attn", "ctx_emb"}
            streams.append(name)
            return SimpleNamespace(name=name)

    class Npu:
        def kernel_set(self, *args):
            return KernelSet()

        def buf(self, *args):
            return Buffer()

    monkeypatch.setitem(sys.modules, "npu_host", SimpleNamespace(Npu=Npu))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(
        Tokenizer=SimpleNamespace(from_file=lambda path: object())))

    def weights(path):
        assert path.name != "text_encoder"
        return SimpleNamespace(get=lambda name: None)

    def packed(cache, model, jobs, *, text_encoder):
        assert not text_encoder
        caches.append(cache.name)
        return {}

    def params(get_dit, get_te):
        assert get_te is None
        return np.zeros(1, np.uint16)

    monkeypatch.setattr(fg, "SafeTensors", weights)
    monkeypatch.setattr(fg, "ensure_packed", packed)
    monkeypatch.setattr(kp, "fill_params", params)
    monkeypatch.setattr(fg, "packed_coder", lambda *args: (np.zeros(1, np.uint8), {}, np.zeros(1, bfloat16), {}))
    ops = [{"phase": phase, "what": name, "stream": name, "set": "gemm", "args": []}
           for phase, name in [("text", "te_qkv"), ("step0", "ctx_emb"), ("step0", "dit")]]
    monkeypatch.setattr(kp, "plan", lambda *args, **kwargs: SimpleNamespace(ops=ops,
        buffers={name: 4 for name in ("PARAMS", "v_QIN", "TF", "DT", "X")},
        vae=SimpleNamespace(buffers={"QIN": SimpleNamespace(H=1, W=1, C=1024)})))
    runner = fg.Runner(tmp_path, tmp_path, 512, 1, False, 1, projected=True)
    assert runner.te is None and caches == ["packed_projected"]
    assert streams == ["dit"]
    assert runner.dispatch_count == 1 and len(runner.ops) == 2


def test_native_capture_waits_and_reads_before_returning_to_the_queue(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "npu_host", SimpleNamespace(COMPLETED="done"))
    events, captures = [], []
    tensors = fixture_tensors()
    bits = tensors["projected_conditioning"].view(np.uint16)
    class Stream:
        name = "ctx_emb"

        def start(self):
            events.append("start")
            return SimpleNamespace(wait=lambda: events.append("wait") or "done")

    def read(dtype, offset, count):
        assert events == ["start", "wait"]
        assert offset == 0 and count == kc.POSITIONS * kc.WIDTH
        events.append("read")
        return bits.reshape(-1).copy()

    original = ("gemm", Stream(), [], "step0", {"what": "ctx_emb"})
    runner = SimpleNamespace(ops=[original], bufs={"X": SimpleNamespace(read=read)})
    restore = replay.capture_first_projection(runner, lambda value: captures.append(value.copy()))
    runner.ops[0][1].start()
    assert events == ["start", "wait", "read"]
    restore()
    assert runner.ops[0] is original
    capture_path = tmp_path / "native.safetensors"
    replay.save_capture(capture_path, captures[0], tensors["input_ids"], 3)
    result = kc.load_projected_conditioning(capture_path)
    assert np.array_equal(result.bits, bits)
    with pytest.raises(FileExistsError):
        replay.save_capture(capture_path, captures[0], tensors["input_ids"], 3)


def test_hardware_gate_does_not_accept_nonfinite_latents(tmp_path):
    runner = SimpleNamespace(set_noise=lambda value: None, run=lambda: {},
        latents=lambda: np.array([[np.nan]], bfloat16).view(np.uint16),
        image=lambda: np.zeros((1, 1, 3), np.uint8))
    with pytest.raises(ValueError, match="nonfinite native latents"):
        replay.snapshot(runner, np.zeros(1, np.uint16), tmp_path, "invalid")
    assert not list(tmp_path.iterdir())
