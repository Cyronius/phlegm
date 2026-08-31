"""Validate a converted model.q4nx.

Checks:
 A. Round-trip: every quantized tensor, dequantized by the verified reader
    (tools/kernel-interp/q4nx.py layout), matches the GGUF source dequant to the
    q4_1/q8 quantization bound.
 B. Ground-truth (when --file given): tensors converted from the synthetic GGUF
    (built out of hf_ref/ originals) match Josh's real model_3LiF.q4nx at the
    quant bound -- proving the whole read->transform->pack->save pipeline.
 C. Structural: names / shapes / dtypes match the q4nx schema.

Usage:
    python validate.py --q4nx synth_out/model.q4nx --gguf synthetic.gguf --file
"""
import argparse, json, struct, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4nx_format import dequant_q4_1_file, dequant_q8_0_file, bf16_u16_to_f32
from convert import dequant_f32, apply_transform, LAYER_MAP, GLOBAL_MAP, deinterleave_qgate

from gguf import GGUFReader


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
    return h, base


def raw_tensor(path, h, base, name):
    o0, o1 = h[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + o0)
        return np.frombuffer(f.read(o1 - o0), np.uint8)


def infer_dims(shape, chunk):
    # I8 tensor [A, B, chunk]; out = A*32, in = B*256
    A, B, _ = shape
    return A * 32, B * 256


def check_roundtrip(q4nx_path, gguf_path):
    print("=== A. Round-trip (converted tensor vs GGUF source dequant) ===")
    h, base = read_header(q4nx_path)
    reader = GGUFReader(gguf_path)
    gt = {t.name: t for t in reader.tensors}
    worst = 0.0
    # map q4nx name -> (gguf name, transform) to reconstruct expected f32
    rev = {}
    for gn, (qn, kind, tr) in GLOBAL_MAP.items():
        rev[qn] = (gn, kind, tr)
    for gn in gt:
        if not gn.startswith("blk."):
            continue
        L = gn.split(".")[1]; suf = gn[len(f"blk.{L}."):]
        if suf in LAYER_MAP:
            qsuf, kind, tr = LAYER_MAP[suf]
            rev[f"model.layer.{L}.{qsuf}"] = (gn, kind, tr)
    for name, meta in h.items():
        if name == "__metadata__" or meta["dtype"] != "I8":
            continue
        chunk = meta["shape"][2]
        out_dim, in_dim = infer_dims(meta["shape"], chunk)
        raw = raw_tensor(q4nx_path, h, base, name)
        gn, kind, tr = rev[name]
        if kind == "q4_experts":
            # per-expert: compare expert-by-expert against source
            src = apply_transform(dequant_f32(gt[gn]), tr)   # [n_e,out,in]
            n_e = src.shape[0]; oe = src.shape[1]
            got = dequant_q4_1_file(raw, out_dim, in_dim)     # [n_e*oe, in]
            e_err = 0.0
            for e in range(n_e):
                a = got[e * oe:(e + 1) * oe]
                b = src[e]
                if np.abs(b).max() > 0:  # skip zero stub experts
                    e_err = max(e_err, float(np.abs(a - b).max()))
            print(f"  {name:48s} experts maxerr={e_err:.4f}")
            worst = max(worst, e_err)
        elif chunk == 8704:
            got = dequant_q8_0_file(raw, out_dim, in_dim)
            src = apply_transform(dequant_f32(gt[gn]), tr)
            err = float(np.abs(got - src).max())
            print(f"  {name:48s} q8 maxerr={err:.4f}")
            worst = max(worst, err)
        else:
            got = dequant_q4_1_file(raw, out_dim, in_dim)
            src = apply_transform(dequant_f32(gt[gn]), tr)
            err = float(np.abs(got - src).max())
            print(f"  {name:48s} q4 maxerr={err:.4f}")
            worst = max(worst, err)
    print(f"  --> worst round-trip maxerr = {worst:.4f} (expect ~<0.08, the q4_1 bound)\n")
    return worst


def check_vs_file(q4nx_path):
    """Compare synthetic-converted layer 0 (HF L0 linear) and layer 1 (HF L3 full-attn)
    against Josh's real model_3LiF.q4nx layer 0 and layer 2 (= orig layer 3)."""
    print("=== B. Ground-truth vs Josh's model_3LiF.q4nx ===")
    FILE = "C:/Users/josha/.flm/models/Qwen3.6-35B-A3B-NPU2/model_3LiF.q4nx"
    if not os.path.exists(FILE):
        print("  (skipped: model_3LiF.q4nx not found)\n"); return
    hf, bf = read_header(FILE)
    hc, bc = read_header(q4nx_path)

    def dq_file(path, h, base, name, od, ind, q8=False):
        raw = raw_tensor(path, h, base, name)
        return (dequant_q8_0_file if q8 else dequant_q4_1_file)(raw, od, ind)

    pairs = [
        # (converted name, file name, out, in)
        ("model.layer.0.linear_attn.qkv_proj.weight", "model.layer.0.linear_attn.qkv_proj.weight", 8192, 2048),
        ("model.layer.0.self_attn.gate_proj.weight",  "model.layer.0.self_attn.gate_proj.weight",  4096, 2048),
        ("model.layer.0.linear_attn.ssm_out_proj.weight", "model.layer.0.linear_attn.ssm_out_proj.weight", 2048, 4096),
        ("model.layer.0.mlp.share_gate_exps_proj.weight", "model.layer.0.mlp.share_gate_exps_proj.weight", 512, 2048),
        ("model.layer.1.self_attn.q_proj.weight", "model.layer.2.self_attn.q_proj.weight", 8192, 2048),
        ("model.layer.1.self_attn.k_proj.weight", "model.layer.2.self_attn.k_proj.weight", 512, 2048),
        ("model.layer.1.self_attn.o_proj.weight", "model.layer.2.self_attn.o_proj.weight", 2048, 4096),
    ]
    worst = 0.0
    for cn, fn, od, ind in pairs:
        got = dq_file(q4nx_path, hc, bc, cn, od, ind)
        ref = dq_file(FILE, hf, bf, fn, od, ind)
        err = float(np.abs(got - ref).max())
        mean = float(np.abs(got - ref).mean())
        print(f"  {cn:46s} vs FILE {fn.split('.')[-3]+'.'+fn.split('.')[-2]:18s} maxerr={err:.4f} mean={mean:.5f}")
        worst = max(worst, err)
    # experts: converted L0 expert 7 vs FILE L0 expert 7
    got = dequant_q4_1_file(raw_tensor(q4nx_path, hc, bc, "model.layer.0.mlp.down_exps_proj.weight"),
                            8 * 2048, 512)[7 * 2048:8 * 2048]
    ref = dequant_q4_1_file(raw_tensor(FILE, hf, bf, "model.layer.0.mlp.down_exps_proj.weight"),
                            256 * 2048, 512)[7 * 2048:8 * 2048]
    err = float(np.abs(got - ref).max())
    print(f"  {'expert7 down (contiguous placement)':46s} {'':26s} maxerr={err:.4f}")
    worst = max(worst, err)
    print(f"  --> worst vs-FILE maxerr = {worst:.4f} (expect ~<0.08; both are quant of same weights)\n")
    return worst


def check_schema(q4nx_path):
    print("=== C. Structural schema (dtypes / chunk sizes) ===")
    h, _ = read_header(q4nx_path)
    ok = True
    for name, meta in h.items():
        if name == "__metadata__":
            continue
        dt, sh = meta["dtype"], meta["shape"]
        if dt == "I8":
            chunk = sh[2]
            if chunk not in (5120, 8704):
                print(f"  [BAD] {name}: I8 chunk {chunk} not 5120/8704"); ok = False
        elif dt not in ("BF16", "F32"):
            print(f"  [BAD] {name}: dtype {dt}"); ok = False
    print(f"  schema {'OK' if ok else 'FAILED'} ({len(h)-('__metadata__' in h)} tensors)\n")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4nx", required=True)
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--file", action="store_true", help="compare vs Josh's model_3LiF.q4nx")
    args = ap.parse_args()
    check_schema(args.q4nx)
    if args.gguf:
        check_roundtrip(args.q4nx, args.gguf)
    if args.file:
        check_vs_file(args.q4nx)
