"""The GPT-OSS source adapter reads only the requested NPU tensor slices."""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "open_kernels"))
from recipes.gptoss_source import (GptOssSource, ONE_EXPERT_CHUNKS, CHUNK,
                                   HEAD_BANDS, required_header, validate_header)  # noqa: E402
from recipes.pack import expert_chunks, fuse_chunks, supertile_perm  # noqa: E402
from export_gptoss_source import export  # noqa: E402
from export_gptoss_head import export as export_head  # noqa: E402


class SparseBytes:
    """A virtual large container; allocate only the slice actually requested."""

    def __init__(self, size):
        self.size = size
        self.reads = []

    def __len__(self):
        return self.size

    def __getitem__(self, part):
        self.reads.append((part.start, part.stop))
        return bytes(part.stop - part.start)


@pytest.fixture
def source():
    fixture = ROOT / "specs/open-engine/tests/fixtures/gptoss_container_header.json"
    # The checked-in inventory intentionally omits offsets; assign virtual
    # contiguous extents without materializing the multi-GB expert payloads.
    tensors = json.loads(fixture.read_text())["tensors"]
    extent = 0
    for tensor in tensors.values():
        size = int(np.prod(tensor["shape"])) * (2 if tensor["dtype"] == "BF16" else 1)
        tensor["data_offsets"] = [extent, extent + size]
        extent += size
    reader = SimpleNamespace(tensors=tensors, data_base=0, mm=SparseBytes(extent))
    return GptOssSource(reader)


def test_source_reads_one_expert_and_one_attention_call(source):
    one = ONE_EXPERT_CHUNKS * CHUNK
    expert_name = "model.layers.0.ffn_gate_up_down_exps.weight"
    offset = source.reader.tensors[expert_name]["data_offsets"][0]
    chunks = source.expert(0, 31, "down")
    assert chunks.shape == (90, 23, CHUNK)
    assert source.reader.mm.reads[-1] == (offset + 31 * one, offset + 32 * one)

    q = source.attention(0, "q", 16)
    assert q.shape == (16, 23, CHUNK)
    qoff = source.reader.tensors["model.layers.0.self_attn.q_proj.weight"]["data_offsets"]
    assert source.reader.mm.reads[-1] == tuple(qoff)


def test_source_exposes_kernel_oriented_constants(source):
    assert source.constant(0, "q_bias").shape == (4096,)
    assert source.constant(0, "sinks").shape == (64,)
    assert source.router_weight(0).shape == (2880, 32)
    assert source.expert_bias(0, 7, "gate").shape == (2880,)
    name = "model.layers.0.mlp.experts.gate_proj_bias"
    start = source.reader.tensors[name]["data_offsets"][0] + 7 * 5760
    assert source.reader.mm.reads[-1] == (start, start + 5760)


def test_qkv_input_selects_only_the_requested_bias_bands(source, monkeypatch):
    from ml_dtypes import bfloat16
    activation = np.full(2880, 2, bfloat16)
    bias = np.arange(4096, dtype=np.float32).astype(bfloat16)
    monkeypatch.setattr(source, "constant", lambda *args: bias.copy())
    packed = source.qkv_input(0, "q", 32, activation)
    assert packed.shape == (3456,)
    assert np.array_equal(packed[:2880], activation)
    assert np.count_nonzero(packed[2880:2944]) == 0
    assert np.array_equal(packed[2944:], bias[1024:1536])
    with pytest.raises(ValueError, match="aligned"):
        source.qkv_input(0, "k", 16, activation)


def test_router_input_appends_the_source_bias(source, monkeypatch):
    from ml_dtypes import bfloat16
    x = np.full(2880, 2, bfloat16)
    bias = np.arange(32, dtype=np.float32).astype(bfloat16)
    monkeypatch.setattr(source, "constant", lambda *args: bias.copy())
    packed = source.router_input(0, x)
    assert packed.shape == (2912,)
    assert np.array_equal(packed[:2880], x)
    assert np.array_equal(packed[2880:], bias)


def test_expert_input_embeds_each_bias_in_the_kernel_chunk(source, monkeypatch):
    from ml_dtypes import bfloat16
    chunks = np.full((90, 23, CHUNK), 9, np.uint8)
    bias = np.arange(2880, dtype=np.float32).astype(bfloat16)
    monkeypatch.setattr(source, "expert", lambda *args: chunks.copy())
    monkeypatch.setattr(source, "expert_bias", lambda *args: bias.copy())
    packed = source.expert_input(0, 7, "gate")
    assert np.array_equal(packed[:, 0, 128:192], bias.view(np.uint8).reshape(90, 64))
    assert np.all(packed[:, 0, 127] == 9)
    assert np.all(packed[:, 0, 192] == 9)
    assert np.all(packed[:, 1, :] == 9)


def test_embedding_and_final_norm_use_bounded_source_reads(source):
    token = 197624
    embedding = source.embedding(token)
    assert embedding.shape == (2880,)
    offset = source.reader.tensors["model.embed_tokens.weight"]["data_offsets"][0]
    assert source.reader.mm.reads[-1] == (offset + token * 5760, offset + (token + 1) * 5760)
    assert source.final_norm().shape == (2880,)
    with pytest.raises(ValueError, match="token id"):
        source.embedding(201088)


def test_complete_gptoss_header_layout_and_disjoint_extents():
    expected = required_header()
    assert len(expected) == 435
    tensors = {}
    offset = 0
    for name, (dtype, shape) in expected.items():
        size = int(np.prod(shape)) * (2 if dtype == "BF16" else 1)
        tensors[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    assert validate_header(tensors) == {"required": 435, "validated": 435, "extra": []}
    broken = {name: dict(value) for name, value in tensors.items()}
    broken["model.layers.23.mlp.router.weight"]["shape"] = [45, 128, 15]
    with pytest.raises(ValueError, match="layout"):
        validate_header(broken)
    overlap = {name: dict(value) for name, value in tensors.items()}
    target = "model.layers.0.input_layernorm.weight"
    size = overlap[target]["data_offsets"][1] - overlap[target]["data_offsets"][0]
    overlap[target]["data_offsets"] = [0, size]
    with pytest.raises(ValueError, match="overlapping"):
        validate_header(overlap)


def test_router_prefill_padding_is_removed_and_rows_are_transposed(source, monkeypatch):
    raw = np.zeros((32, 2944), np.float32)
    raw[7, 123] = 2.5
    raw[:, 2880:] = 99
    from ml_dtypes import bfloat16
    monkeypatch.setattr(source, "_bytes", lambda *args, **kwargs: raw.astype(bfloat16).tobytes())
    weight = source.router_weight(0)
    assert weight.shape == (2880, 32)
    assert float(weight[123, 7]) == 2.5
    assert np.count_nonzero(weight) == 1


def test_source_preserves_container_supertile_and_expert_role_order(source, monkeypatch):
    q_raw = np.zeros((128 * 23, CHUNK), np.uint8)
    q_raw[:, 0] = np.arange(len(q_raw), dtype=np.uint16) % 256
    monkeypatch.setattr(source, "_bytes", lambda *args, **kwargs: q_raw.tobytes())
    selected = source.attention(0, "q", 32)
    perm = supertile_perm(128, 23, 4)[32:48]
    assert np.array_equal(selected.reshape(-1, CHUNK)[:, 0], q_raw[perm.reshape(-1), 0])

    expert_raw = np.zeros((ONE_EXPERT_CHUNKS, CHUNK), np.uint8)
    expert_raw[:, 0] = np.arange(len(expert_raw), dtype=np.uint16) % 256
    monkeypatch.setattr(source, "_bytes", lambda *args, **kwargs: expert_raw.tobytes())
    selected = source.expert(0, 2, "up")
    order = expert_chunks(23, 23, 4)[1, :90].reshape(-1)
    assert np.array_equal(selected.reshape(-1, CHUNK)[:, 0], expert_raw[order, 0])


def test_source_rejects_wrong_shape_and_truncated_payload(source):
    name = "model.layers.0.self_attn.q_proj.weight"
    tensor = source.reader.tensors[name]
    original = tensor["shape"]
    tensor["shape"] = [31, 23, 4, CHUNK]
    with pytest.raises(ValueError, match="layout"):
        source.attention(0, "q", 0)
    tensor["shape"] = original
    source.reader.mm.size = tensor["data_offsets"][1] - 1
    with pytest.raises(ValueError, match="truncated"):
        source.attention(0, "q", 0)


def test_source_rejects_unsupported_call_indices(source):
    with pytest.raises(ValueError, match="aligned"):
        source.attention(0, "q", 1)
    with pytest.raises(ValueError, match="expert id"):
        source.expert(0, 32, "gate")
    with pytest.raises(ValueError, match="layer"):
        source.constant(24, "sinks")
    with pytest.raises(ValueError, match="head band"):
        source.head_band(HEAD_BANDS)


def test_head_band_fuses_half_chunks_in_q4_kernel_order(source, monkeypatch):
    half = np.zeros((23, 2, CHUNK), np.uint8)
    half[:, :, 0] = np.arange(46, dtype=np.uint8).reshape(23, 2)
    monkeypatch.setattr(source, "_bytes", lambda *args, **kwargs: half.tobytes())
    full = source.head_band(31)
    assert full.shape == (24, 5120)
    for kt in range(11):
        for rb in range(2):
            expected = fuse_chunks(half[2 * kt, rb], half[2 * kt + 1, rb])[0]
            assert np.array_equal(full[2 * kt + rb], expected)
    assert np.array_equal(full[22], fuse_chunks(half[22, 0], np.zeros(CHUNK, np.uint8))[0])
    assert np.array_equal(full[23], fuse_chunks(half[22, 1], np.zeros(CHUNK, np.uint8))[0])


def test_head_band_reads_only_its_own_container_slice(source):
    band = 17
    source.head_band(band)
    start = source.reader.tensors["lm_head.weight"]["data_offsets"][0] + band * 23 * 2 * CHUNK
    assert source.reader.mm.reads[-1] == (start, start + 23 * 2 * CHUNK)


def test_exported_layer_has_all_attention_calls_and_recorded_bytes(source, tmp_path):
    out = tmp_path / "layer"
    manifest = export(source, 0, out)
    assert len(manifest["files"]) == 25
    assert (out / "attention/q-112.bin").stat().st_size == 16 * 23 * CHUNK
    assert (out / "constants/router_weight.bin").stat().st_size == 2880 * 32 * 2
    assert manifest["files"]["attention/q-112.bin"]["bytes"] == 16 * 23 * CHUNK
    with pytest.raises(FileExistsError):
        export(source, 0, out)
    with pytest.raises(ValueError, match="distinct"):
        export(source, 0, tmp_path / "bad", (3, 3))
    assert not (tmp_path / "bad").exists()


def test_head_export_streams_bands_and_records_digest(tmp_path):
    class Source:
        def head_band(self, band):
            result = np.zeros((24, 5120), np.uint8)
            result[0, 0] = band + 1
            return result

    out = tmp_path / "head"
    report = export_head(Source(), out, bands=2)
    data = (out / "head.bin").read_bytes()
    assert len(data) == 2 * 24 * 5120
    assert data[0] == 1 and data[24 * 5120] == 2
    assert report["head"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert not report["complete"]
