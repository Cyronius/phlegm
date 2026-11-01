"""One Qwen Image 2.1 transformer block on an edit's condition-latent layout, on the NPU.

The joint sequence is the published order [text, condition image, text, target image]
(``qwen21_edit_layout_reference.py``). Row-wise streams (norm, QKV, out-projection,
gated residuals, MLP) run on 512-row chunks grouped by modulation: text and
condition rows use the t = 0 row, target rows the sampled one (``causal_condition``).
The q/k stream runs over the joint rows with ``cond_row`` (the condition block's start),
which gives the published 3-axis positions including the after-block gap. Attention
follows ``qwen21_attention_plan.segment_passes``: one causal pass over every row up to
the last text row (its text rows kept), then one bidirectional prefix pass per image
block. The host only stages rows between streams.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from open_kernels.qwen21_attention_plan import segment_passes
from open_kernels.qwen21_block_run import HarnessStage, bf, params, qk_params, tensor
from open_kernels.qwen21_weight_layout import HIDDEN, HEAD_DIM

CHUNK, MLP = 512, 12288


def pad_rows(x: np.ndarray, rows: int) -> np.ndarray:
    out = np.zeros((rows, x.shape[1]), bfloat16)
    out[:x.shape[0]] = x
    return out


def run(args) -> None:
    ref = args.reference.resolve()
    layout = json.loads((ref / "layout.json").read_text())
    segments = tuple((k, n) for k, n in layout["segments"])
    joint = layout["joint"]
    target_start = joint - segments[-1][1]
    x = bf(np.fromfile(ref / "block0_input.f32", "<f4").reshape(joint, HIDDEN))
    mod = np.fromfile(ref / "modulation.f32", "<f4").reshape(2, 4, HIDDEN)
    mod_t, mod_0 = bf(mod[0]), bf(mod[1])
    kernels = {k: Path(v) for k, v in json.loads(args.kernels.read_text()).items()}
    packed = json.loads((args.weights / "packed_weights.json").read_text())
    weights = {n: args.weights / packed["slots"][f"block.0.{n}"]["file"]
               for n in ("qkv", "attn_out", "mlp_in", "mlp_out")}
    wq = tensor(args.model, "transformer_blocks.0.attn.norm_q.weight")
    wk = tensor(args.model, "transformer_blocks.0.attn.norm_k.weight")
    if wq.shape != (HEAD_DIM,) or wk.shape != (HEAD_DIM,):
        raise ValueError("unexpected Qwen q/k norm width")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    def stage(name, *keys):
        return HarnessStage(out, name, {k: kernels[k] for k in keys}, args.harness)

    # chunks: (global rows, modulation)
    groups = [(np.arange(0, target_start), mod_0), (np.arange(target_start, joint), mod_t)]
    chunks = []
    for rows, m in groups:
        for i in range(0, len(rows), CHUNK):
            chunks.append((rows[i:i + CHUNK], m))

    s = stage("01-qkv", "norm", "qkv")
    s.buf("w_qkv", path=weights["qkv"])
    for i, (rows, m) in enumerate(chunks):
        s.buf(f"x{i}", pad_rows(x[rows], CHUNK))
        s.buf(f"p{i}", params(np.zeros(HIDDEN), m[0]))
        s.buf(f"d{i}", shape=(HIDDEN,))
        s.buf(f"n{i}", shape=(CHUNK, HIDDEN))
        s.buf(f"qkv{i}", shape=(CHUNK, 3 * HIDDEN))
        s.run("norm", f"x{i}", f"d{i}", f"p{i}", f"n{i}", f"d{i}")
        s.run("qkv", f"n{i}", "w_qkv", f"qkv{i}")
        s.dump(f"qkv{i}", (CHUNK, 3 * HIDDEN))
    got = s.execute()
    qkv_rows = (joint + CHUNK - 1) // CHUNK * CHUNK
    qkv = np.zeros((qkv_rows, 3 * HIDDEN), bfloat16)
    for i, (rows, _) in enumerate(chunks):
        qkv[rows] = got[f"qkv{i}"][:len(rows)]

    s = stage("02-qk", "qk")
    s.buf("qkv", qkv)
    s.buf("p", qk_params(wq, wk))
    s.buf("q", shape=(qkv_rows, HIDDEN))
    s.buf("k", shape=(qkv_rows, HIDDEN))
    s.run("qk", "qkv", "qkv", "p", "q", "k")
    s.dump("q", (qkv_rows, HIDDEN))
    s.dump("k", (qkv_rows, HIDDEN))
    rot = s.execute()
    q, k, v = rot["q"], rot["k"], qkv[:, 2 * HIDDEN:]

    passes = segment_passes(segments)
    names = {"text": "fa_text", "image.0": "fa_cond", "image.1": "fa_target"}
    s = stage("03-attention", *names.values())
    for p in passes:
        n = p.kind.replace(".", "")
        qs = pad_rows(q[p.q_start:p.q_end] if p.kind != "text" else q[:p.q_end], p.q_rows)
        s.buf(f"q_{n}", qs)
        s.buf(f"k_{n}", pad_rows(k[:p.valid_len], p.kv_rows))
        s.buf(f"v_{n}", pad_rows(v[:p.valid_len], p.kv_rows))
        s.buf(f"o_{n}", shape=qs.shape)
        s.run(names[p.kind], f"q_{n}", f"k_{n}", f"v_{n}", f"o_{n}")
        s.dump(f"o_{n}", qs.shape)
    got = s.execute()
    attention = np.zeros((joint, HIDDEN), bfloat16)
    for p in passes:
        o = got[f"o_{p.kind.replace('.', '')}"]
        base = 0 if p.kind == "text" else p.q_start
        for a, b in p.keep:
            attention[a:b] = o[a - base:b - base]

    s = stage("04-residual-mlp", "attn_out", "res", "mlp_in", "mlp_out")
    for name in ("attn_out", "mlp_in", "mlp_out"):
        s.buf(f"w_{name}", path=weights[name])
    for i, (rows, m) in enumerate(chunks):
        s.buf(f"x{i}", pad_rows(x[rows], CHUNK))
        s.buf(f"a{i}", pad_rows(attention[rows], CHUNK))
        s.buf(f"pf{i}", params(m[1], np.zeros(HIDDEN), m[2]))
        s.buf(f"ps{i}", params(m[3], np.zeros(HIDDEN), np.zeros(HIDDEN)))
        for name, shape in (("ao", (CHUNK, HIDDEN)), ("n2", (CHUNK, HIDDEN)), ("x1", (CHUNK, HIDDEN)),
                            ("mlp", (CHUNK, 2 * MLP)), ("mo", (CHUNK, HIDDEN)),
                            ("un", (CHUNK, HIDDEN)), ("x2", (CHUNK, HIDDEN))):
            s.buf(f"{name}{i}", shape=shape)
        s.run("attn_out", f"a{i}", "w_attn_out", f"ao{i}")
        s.run("res", f"x{i}", f"ao{i}", f"pf{i}", f"n2{i}", f"x1{i}")
        s.run("mlp_in", f"n2{i}", "w_mlp_in", f"mlp{i}")
        s.run("mlp_out", f"mlp{i}", "w_mlp_out", f"mo{i}")
        s.run("res", f"x1{i}", f"mo{i}", f"ps{i}", f"un{i}", f"x2{i}")
        s.dump(f"x2{i}", (CHUNK, HIDDEN))
    got = s.execute()
    result = np.zeros((joint, HIDDEN), bfloat16)
    for i, (rows, _) in enumerate(chunks):
        result[rows] = got[f"x2{i}"][:len(rows)]
    result.tofile(out / "native-block0.bf16")
    attention.tofile(out / "native-attention.bf16")
    q[:joint].tofile(out / "native-q.bf16")

    ref_out = np.fromfile(ref / "block0_output.f32", "<f4").reshape(joint, HIDDEN).astype(np.float64)
    nat = result.astype(np.float64)

    def metric(a, b):
        return {"relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                "cosine": float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))}

    report, offset = {"all": metric(nat, ref_out)}, 0
    for i, (kind, n) in enumerate(segments):
        report[f"{i}:{kind}"] = metric(nat[offset:offset + n], ref_out[offset:offset + n])
        offset += n
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, r in report.items():
        print(f"{name:10s} rel L2 {r['relative_l2'] * 100:7.3f}%  cos {r['cosine']:.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True, help="qwen21_edit_layout_reference output")
    p.add_argument("--kernels", type=Path, required=True,
                   help="JSON: norm, res, qkv, attn_out, mlp_in, mlp_out, qk, fa_text, fa_cond, fa_target")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--harness", type=Path, default=(
        Path(__file__).resolve().parents[1] / "target/npu-dev/harness-build/run_kernel"))
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
