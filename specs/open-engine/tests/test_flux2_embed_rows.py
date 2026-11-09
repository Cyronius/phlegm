"""FLUX.2 [klein] XT: the prompt's embedding rows reach the NPU as bf16 bits whatever
dtype the text encoder's checkpoint stores (the published BF16, or an F16 re-save such as
an abliterated encoder)."""
import json
import struct

import numpy as np
from ml_dtypes import bfloat16

import flux2_native_generate as fg
import klein_pipeline as kp


def write_table(path, table, dtype):
    raw = table.astype(bfloat16).view(np.uint16) if dtype == "BF16" else table.astype(np.float16)
    data = raw.tobytes()
    hdr = json.dumps({"model.embed_tokens.weight": {
        "dtype": dtype, "shape": list(table.shape), "data_offsets": [0, len(data)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(hdr)) + hdr + data)


def test_bf16_and_f16_tables_give_the_same_bf16_rows(tmp_path):
    rng = np.random.default_rng(0)
    # f16-representable values that are also bf16-exact after rounding through f32
    table = rng.standard_normal((16, 40)).astype(np.float16).astype(np.float32)
    ids = np.array([3, 0, 15, 3, 7])
    rows = {}
    for dtype in ("BF16", "F16"):
        d = tmp_path / dtype
        d.mkdir()
        write_table(d / "model.safetensors", table, dtype)
        rows[dtype] = fg.embed_rows(fg.SafeTensors(d), ids)
    want = table[ids].astype(bfloat16).view(np.uint16)
    for dtype, x in rows.items():
        assert x.dtype == np.uint16 and x.shape == (len(ids), kp.TE_PAD), dtype
        assert np.array_equal(x[:, :40], want), dtype
        assert not x[:, 40:].any(), dtype
