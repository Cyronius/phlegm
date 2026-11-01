"""One HY V3 layer through hy_att, the host route and hy_moe on the NPU, from the GGUF, at the position the
streams were compiled for (1: one cached row, the new row at row 1; slot j = pool expert j), against the
fp32 GGUF reference (model/hyv3_gguf_reference.py).

    python designs/hyv3/check_layer.py --driver run_kernel --att BUILD --moe BUILD --gguf model.gguf \
        --config config.json --ids-file raw.ids --layer 1 --out DIR

The cached row 0 is token 0's K / V, computed here in fp32 from the GGUF and rounded to bf16 as the
kernel stores it. Each stage is reported against the reference (the model's numbers: the gap includes
every bf16 activation boundary; the q4_1 pools are exact), and the FFN again against the reference fed
the NPU's own res / xm and the same experts (the kernels' arithmetic alone). The route runs on the host
exactly as the runtime's hy_route does (the reference's `route`). The dense layer 0 is the same two
programs with its MLP's nine 768-wide slices in slots 0..8 at weight 1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
OK = HERE.parents[1]
sys.path.insert(0, str(OK))
sys.path.insert(0, str(OK / "model"))
from flint_gguf_reference import Gguf, rms, rope, silu  # noqa: E402
from hyv3_gguf_reference import attention, forward, rope_inv_freq, route  # noqa: E402
from recipes import hyv3 as FR  # noqa: E402
from recipes import flint_pack as P  # noqa: E402   (fl_q4's byte law: the same pack op)
from recipes.load import spec_from_model_dir  # noqa: E402
from recipes.pack import ptab  # noqa: E402


def stats(name, got, ref):
    got, ref = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    rel = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30)
    cos = float(got @ ref / max(np.linalg.norm(got) * np.linalg.norm(ref), 1e-30))
    mx = float(np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30))
    print(f"  {name:28s} rel L2 {rel:.3e}  cos {cos:.9f}  max|d|/max|ref| {mx:.3e}", flush=True)
    return {"rel_l2": rel, "cos": cos, "max_rel": mx}


def q4(g: Gguf, name: str) -> np.ndarray:
    t = g.tensors[name]
    cols = t["dims"][0]
    rows = int(np.prod(t["dims"][1:]))
    q41 = t["gguf_type"] == 3
    o = g.base + t["offset"]
    return P.blocks(g.mm[o:o + rows * cols // 32 * (20 if q41 else 18)], rows, cols, q41)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("att", "moe", "gguf", "config", "ids-file", "out", "driver"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    spec = spec_from_model_dir(Path(a.config).parent)
    R = FR.recipe(spec)
    L, G = R.layout, R.geo
    l, pos = a.layer, 1
    kind = FR.kinds(spec)[l]
    dense = kind == FR.KIND_DENSE
    PL = L
    POOL = L.POOL_BYTES
    g = Gguf(a.gguf)
    ids = [int(t) for t in Path(a.ids_file).read_text().replace(",", " ").split()][:2]
    p = f"blk.{l}."
    H, KVH, hd, eps = G.NH, G.KVH, G.HD, G.EPS
    inv = rope_inv_freq(cfg)

    # ---- the reference for tokens 0 and 1 through layer l (layers before l in fp32 too)
    x = g.rows("token_embd.weight", ids)
    if l > 0:
        _, _, _, x = forward(g, cfg, ids, layers=l)
    hn = rms(x, g.array(p + "attn_norm.weight"), eps)
    q = rms((hn @ g.array(p + "attn_q.weight").T).reshape(2, H, hd), g.array(p + "attn_q_norm.weight"), eps)
    k = rms((hn @ g.array(p + "attn_k.weight").T).reshape(2, KVH, hd), g.array(p + "attn_k_norm.weight"), eps)
    v = (hn @ g.array(p + "attn_v.weight").T).reshape(2, KVH, hd)
    q, k = rope(q, np.arange(2), inv), rope(k, np.arange(2), inv)
    att = attention(q, k, v)
    o = att.reshape(2, H * hd) @ g.array(p + "attn_output.weight").T
    res_ref = (x + o)[pos]
    xm_ref = rms(res_ref, g.array(p + "ffn_norm.weight"), eps)

    driver = Path(a.driver).resolve()
    a.att, a.moe = str(Path(a.att).resolve()), str(Path(a.moe).resolve())

    def bf(vec, n):
        o = np.zeros(n, bfloat16)
        o[:len(vec)] = vec
        return o
    pool = np.lib.format.open_memmap(out / "pool.npy", "w+", np.uint8, (POOL,))
    pool_bin = out / "pool.bin"
    for role, dst, rows, K in (("attn_q", PL.POOL_Q, G.QW, G.HID), ("attn_k", PL.POOL_K, G.KVW, G.HID),
                               ("attn_v", PL.POOL_V, G.KVW, G.HID), ("attn_output", PL.POOL_O, G.HID, G.QW)):
        t = P.matrix(q4(g, p + role + ".weight"), rows, K)
        pool[dst:dst + t.size] = t
    consts = np.zeros(L.CD_BYTES, np.uint8)
    for vec, n, off in ((g.array(p + "attn_norm.weight"), G.HID, L.CD_LNW),
                        (g.array(p + "ffn_norm.weight"), G.HID, L.CD_POSTLN),
                        (g.array(p + "attn_q_norm.weight"), G.HD, L.CD_META),
                        (g.array(p + "attn_k_norm.weight"), G.HD, L.CD_META + 2 * G.HD)):
        b = bf(vec, n).view(np.uint8)
        consts[off:off + b.size] = b
    kv = np.zeros(2 * L.KV_ROW, np.uint8)
    kv[:L.KV_ROW] = np.concatenate([k[0].reshape(-1), v[0].reshape(-1)]).astype(bfloat16).view(np.uint8)
    ptab_b = ptab(L.MAX_CTX, G.ROT, spec.rope_theta, L.PTAB_ROW, spec.rope_inv_freq(), 0, 1.0)
    xin = x[pos].astype(np.float32)
    for name, arr in (("consts", consts), ("kv", kv), ("ptab", ptab_b), ("xres", xin)):
        (out / f"{name}.bin").write_bytes(np.ascontiguousarray(arr).tobytes())

    def drive(name, lines):
        cfg_path = out / f"{name}.cfg"
        cfg_path.write_text("\n".join(lines) + "\n")
        r = subprocess.run([str(driver), str(cfg_path)], capture_output=True, text=True)
        print(r.stdout.strip(), flush=True)
        if r.returncode:
            raise RuntimeError(f"{name}: {r.stdout}\n{r.stderr}")

    def pool_file():
        pool.flush()
        with open(pool_bin, "wb") as f:                       # the .npy header off, sparse where zero
            f.truncate(POOL)
        mm = np.memmap(pool_bin, np.uint8, "r+", shape=(POOL,))
        nz = np.flatnonzero(pool.reshape(-1, 1 << 20).any(axis=1))
        for i in nz:
            mm[i << 20:(i + 1) << 20] = pool[i << 20:(i + 1) << 20]
        mm.flush()

    pool_file()
    drive("att", ["device", f"xclbin A {Path(a.att) / 'final.xclbin'}", f"kernelx ka A {Path(a.att) / 'insts.bin'}",
                  f"buf pool {POOL} pool.bin", f"buf xres {G.HID * 4} xres.bin", f"buf consts {L.CD_BYTES} consts.bin",
                  f"buf kv {L.KV_BYTES} kv.bin", f"buf act {L.AD_BYTES}", f"buf ptab {L.PTAB_BYTES} ptab.bin",
                  "run ka pool xres consts kv act ptab", f"dump act act_att.bin {L.AD_BYTES}",
                  f"dump kv kv_att.bin {2 * L.KV_ROW}"])
    actb = np.fromfile(out / "act_att.bin", np.uint8)
    res = actb[L.AD_RES:L.AD_RES + G.HID * 4].view(np.float32).copy()
    xm = actb[L.AD_XM:L.AD_XM + G.HID * 2].view(bfloat16).astype(np.float32)
    krow = np.fromfile(out / "kv_att.bin", np.uint8)[L.KV_ROW:].view(bfloat16).astype(np.float32)
    rep = {"layer": l, "position": pos, "ids": ids, "kind": kind}
    print(f"attention half, layer {l} ({kind}) (vs fp32 reference):", flush=True)
    rep["k"] = stats("new K row", krow[:G.KVW], k[1].reshape(-1))
    rep["v"] = stats("new V row", krow[G.KVW:], v[1].reshape(-1))
    og = actb[L.AD_OG:L.AD_OG + G.QW * 2].view(bfloat16).astype(np.float32)
    rep["og"] = stats("attention out (og)", og, att[pos].reshape(-1))
    rep["res"] = stats("res = x + o", res, res_ref)
    rep["xm"] = stats("xm", xm, xm_ref)

    # ---- the route (hy_route) from the NPU's xm, and the slots' weights into the pool
    ff = G.FF
    nslot = G.SLOTS
    if dense:
        slots = [("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", s) for s in range(nslot)]
        w = np.ones(nslot, np.float32)
        top = ref_top = np.arange(nslot)
    else:
        wr, br, sc = g.array(p + "ffn_gate_inp.weight"), g.array(p + "exp_probs_b"), FR.router_scale(spec)
        top, w8 = route(xm[None], wr, br, G.TOPK, sc)
        ref_top, ref_w = route(xm_ref[None], wr, br, G.TOPK, sc)
        top, w8, ref_top, ref_w = top[0], w8[0], ref_top[0], ref_w[0]
        rep["experts"], rep["ref_experts"], rep["weights"] = top.tolist(), ref_top.tolist(), w8.tolist()
        print(f"route: experts {top.tolist()} (reference {ref_top.tolist()}), weights {np.round(w8, 4).tolist()}")
        w = np.concatenate([w8, np.ones(G.SHARED, np.float32)])
        slots = [("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", int(e)) for e in top]
        slots += [("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight", h)
                  for h in range(G.SHARED)]
    wv = np.zeros(FR.WV, np.float32)
    wv[:nslot] = w
    actb[L.AD_W:L.AD_W + 4 * FR.WV] = wv.view(np.uint8)
    actb.tofile(out / "act_route.bin")
    blk = {}
    for nm in {s[i] for s in slots for i in range(3)}:
        blk[nm] = q4(g, p + nm)
    dw = {}
    for s_, (gn, un, dn, e) in enumerate(slots):
        shared = gn.endswith("shexp.weight") or gn == "ffn_gate.weight"
        r0 = e * ff
        gate = P.matrix(blk[gn], ff, G.HID, row0=r0)
        up = P.matrix(blk[un], ff, G.HID, row0=r0)
        ug = P.slot_ug(gate, up, G.HID)
        pool[s_ * L.EXPERT_UG:s_ * L.EXPERT_UG + ug.size] = ug
        if shared:      # down's columns [e ff, e ff + ff) of a [HID, 2 ff or DFF] matrix
            down = P.matrix(blk[dn], G.HID, ff, col0=e * ff)
        else:           # down's rows [e HID, e HID + HID) of the [E HID, ff] flattened expert tensor
            down = P.matrix(blk[dn], G.HID, ff, row0=e * G.HID)
        o0 = PL.POOL_DOWN + s_ * L.UP_BYTES
        pool[o0:o0 + down.size] = down
        if shared:
            Wg, Wu = g.array(p + gn)[r0:r0 + ff], g.array(p + un)[r0:r0 + ff]
            Wd = g.array(p + dn)[:, e * ff:(e + 1) * ff]
        else:
            Wg, Wu, Wd = g.expert(p + gn, e), g.expert(p + un, e), g.expert(p + dn, e)
        dw[s_] = (Wg, Wu, Wd)
    pool_file()
    drive("moe", ["device", f"xclbin M {Path(a.moe) / 'final.xclbin'}", f"kernelx km M {Path(a.moe) / 'insts.bin'}",
                  f"buf pool {POOL} pool.bin", f"buf xres {G.HID * 4} xres.bin",
                  f"buf act {L.AD_BYTES} act_route.bin", "run km pool xres act",
                  f"dump xres xres_moe.bin {G.HID * 4}", f"dump act act_moe.bin {L.AD_BYTES}"])
    y = np.fromfile(out / "xres_moe.bin", np.float32)
    h = np.fromfile(out / "act_moe.bin", np.uint8)[L.AD_H:L.AD_H + nslot * G.H_STRIDE].view(np.float32)
    h = h.reshape(nslot, G.H_STRIDE // 4)[:, :ff]

    def ffn(res_in, xm_in, wts):
        acc = np.zeros(G.HID, np.float64)
        hs = []
        for s_ in range(nslot):
            Wg, Wu, Wd = dw[s_]
            hh = wts[s_] * silu(Wg @ xm_in) * (Wu @ xm_in)
            hs.append(hh)
            acc += Wd.astype(np.float64) @ hh.astype(np.float64)
        return res_in + acc, hs
    y_cond, h_cond = ffn(res, xm, w)
    print("FFN half (vs the reference fed the NPU's res / xm and the same slots):", flush=True)
    for s_ in range(nslot):
        rep[f"h{s_}"] = stats(f"h_{s_} ({slots[s_][0].split('.')[0]} {slots[s_][3]})", h[s_], h_cond[s_])
    rep["out_cond"] = stats("layer out", y, y_cond)
    if dense or list(top) == list(ref_top):
        y_ref, _ = ffn(res_ref, xm_ref, w if dense else np.concatenate([ref_w, np.ones(G.SHARED)]))
        print("FFN half (vs the fp32 reference):", flush=True)
        rep["out"] = stats("layer out", y, y_ref)
    else:
        print("FFN half: the NPU's route differs from the reference's; no fp32 comparison of the layer out")
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
