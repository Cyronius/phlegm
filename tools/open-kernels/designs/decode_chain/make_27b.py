r"""Josh's pruned Qwen3.6-27B-A2.8B (30 layers, full_attention_interval=3) through
the open kernels: one decode step at position 0 (empty states/cache), all 30
layers + final norm + lm_head, as one driver config. Oracle: the HF-faithful CPU
replica (decode_step.py) on the same q4nx -- FLM cannot be the oracle for an
interval-3 model (it skips the full-attention block, see the plan's Finding).

Weights come from the q4nx via tools/kernel-interp/build_pools.py (the same pool
/pack/side layouts the resident engine builds), sliced per kernel. Run from
tools/kernel-interp (it imports decode_step, which loads MODEL_Q4NX):

    cd tools/kernel-interp && MODEL_Q4NX=/mnt/c/Users/josha/.flm/models/Qwen3.6-27B-A2.8B-open/model.q4nx \
        python .../decode_chain/make_27b.py [--layers N] [--token T]
    ATTN_POS=0 python build_design.py designs/attn/attn.py designs/attn/build_pos0   (once)
    open-qwen-npu npu designs/decode_chain/run_27b.cfg ; python compare_27b.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).parent
KI = Path("/mnt/c/code/phlegm/tools/kernel-interp")
sys.path.insert(0, str(KI))
MODEL_DIR = "/mnt/c/Users/josha/.flm/models/Qwen3.6-27B-A2.8B-open"
os.environ.setdefault("MODEL_Q4NX", f"{MODEL_DIR}/model.q4nx")
os.chdir(KI)
import decode_step as DS  # noqa: E402
import build_pools as BP  # noqa: E402
from q4nx import bf16_to_f32  # noqa: E402
sys.path.insert(0, str(HERE.parent / "lin_layer"))
import layout as LL  # noqa: E402  (lin_a / lin_c byte layouts)

D = "C:/code/phlegm/tools/open-kernels/designs"
OUT = f"{D}/decode_chain/w27"
POOLS = "C:/code/FastFlowLM/npu-engine/m3out/l30_27b"      # `open-qwen-npu l30-build` output (pools.rs layout)
WDIR = HERE / "w27"
S = 163_840
NE = 8


def bf(x):
    return np.asarray(x, np.float32).astype(bfloat16).astype(np.float64)


def wr(name, arr):
    a = np.ascontiguousarray(arr)
    (WDIR / name).write_bytes(a.tobytes())


def routing(m, l, x_res):
    postln = m.bf16(f"model.layer.{l}.post_attention_layernorm.weight")
    xm = bf(DS.F.rms(x_res) * postln)
    lg = xm @ m.bf16(f"model.layer.{l}.moe_router.weight").astype(np.float64)
    p = np.exp(lg - lg.max()); p /= p.sum()
    top = np.argsort(-p, kind="stable")[:NE]
    prev = WDIR / f"y_rout{l}.bin"
    if prev.is_file():
        got = np.fromfile(prev, np.float32)[256:264].view(np.int32)
        if got.tolist() != top.tolist():
            print(f"layer {l}: NPU routing {got.tolist()} != predicted {top.tolist()}; using the NPU's")
            top = got.astype(np.int64)
    return top


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None, help="only the first N layers (+ final norm/lm_head)")
    ap.add_argument("--token", type=int, default=248045)
    ap.add_argument("--cfg-only", action="store_true", help="only rewrite run_27b.cfg (weights/refs from a previous run)")
    a = ap.parse_args()
    WDIR.mkdir(exist_ok=True)
    cfgj = json.load(open(f"{MODEL_DIR}/config.json"))
    NL = a.layers or cfgj["num_hidden_layers"]
    INT = cfgj["full_attention_interval"]
    m = DS.m
    full = {l: ((l + 1) % INT == 0) for l in range(NL)}
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    x = bf16_to_f32(np.frombuffer(m.mm[base + a.token * 4096: base + (a.token + 1) * 4096], dtype=np.uint16)).astype(np.float64)
    wr("xres0.bin", x.astype(np.float32))
    wr("zero.bin", np.zeros(2048, np.float32))
    wr("zstate.bin", np.zeros(3 * 8192, bfloat16))
    wr("zS.bin", np.zeros(32 * 128 * 128, np.float32))
    wr("zkv.bin", np.zeros(3145728, np.uint8))
    normw = m.bf16("model.norm.weight")
    wr("normw.bin", normw.astype(np.float32).astype(bfloat16))
    freqs = (1e7) ** (-np.arange(32) / 32)
    meta_cs = np.concatenate([np.cos(0 * freqs), np.sin(0 * freqs)]).astype(np.float32)

    # ---- replica + per-layer routing, and weights per layer
    cs = np.zeros((3, 8192)); S0 = np.zeros((32, 128, 128))
    top = {}
    xr = x.copy()
    for l in range(NL if not a.cfg_only else 0):
        if full[l]:
            xa, _, _ = DS.attn_decode(l, xr.copy(), np.zeros((0, 2, 256)), np.zeros((0, 2, 256)), 0)
        else:
            xa, _, _ = DS.linear_decode(l, xr.copy(), cs.copy(), S0.copy())
        top[l] = routing(m, l, xa)
        xr = DS.moe_decode(l, xa)
        wr(f"ref_res{l}.bin", xr.astype(np.float32))
        pool = np.frombuffer(BP.build_layer_pool(m, l, full[l]), np.uint8)
        pack = np.frombuffer(BP.build_pack(m, l), np.uint8)
        side = np.frombuffer(BP.build_side(m, l, full[l]), np.uint8)
        wr(f"lnw{l}.bin", pack[0:4096]); wr(f"postln{l}.bin", pack[4096:8192])
        wr(f"sgw{l}.bin", pack[8192:12288]); wr(f"rw{l}.bin", pack[12288:12288 + 1048576])
        # (the experts stream straight out of the resident layer pool: the
        # driver's `moeroute` points the kernel's fills at the router's choice)
        if not full[l]:
            # (qkv / z stream straight out of the resident pool: lin_a reads them at their pool offsets)
            wr(f"wout{l}.bin", side[328_192:328_192 + 10_485_760])
            nwp = np.zeros(2048, bfloat16); nwp[:128] = side[65536:65536 + 256].view(bfloat16)
            wr(f"nw{l}.bin", nwp)
            sb = np.zeros(335872, np.uint8)
            sb[4096:4096 + 131072] = side[66048:66048 + 131072]
            sb[135168:135168 + 131072] = side[197120:197120 + 131072]
            small = np.zeros(1024, np.float32)
            small[:32] = side[65792:65792 + 128].view(np.float32); small[32:64] = side[65920:65920 + 128].view(np.float32)
            sb[266240:266240 + 4096] = small.view(np.uint8)
            convw = side[0:65536].view(bfloat16).reshape(4, 8192)
            sb[270336:270336 + 65536] = convw.reshape(4, 8, 1024).transpose(1, 0, 2).reshape(-1).view(np.uint8)
            wr(f"side{l}.bin", sb)
        else:
            wr(f"wq{l}.bin", pool[505_282_560:505_282_560 + 5_242_880])
            wr(f"wk{l}.bin", pool[510_525_440:510_525_440 + 655_360])
            wr(f"wv{l}.bin", pool[511_180_800:511_180_800 + 655_360])
            wr(f"wgate{l}.bin", pool[511_836_160:511_836_160 + 5_242_880])
            wr(f"wo{l}.bin", pool[517_079_040:517_079_040 + 5_242_880])
            meta = np.zeros(2048, np.uint8)
            meta[512:1024] = side[128:640]; meta[1024:1536] = side[640:1152]
            meta[1536:1792] = meta_cs.view(np.uint8)
            wr(f"meta{l}.bin", meta)
        print(f"layer {l} {'FULL' if full[l] else 'lin '} top8={top[l].tolist()}", flush=True)
    if not a.cfg_only:
        hn = (DS.F.rms(xr) * normw).astype(np.float32)
        logits_ref = m.lmhead_logits(hn)
        wr("ref_logits.bin", logits_ref.astype(np.float32))
        if not (WDIR / "lm27.bin").is_file():
            (WDIR / "lm27.bin").write_bytes(BP.build_lmhead_pool(m))
    ref_argmax = int(np.fromfile(WDIR / "ref_logits.bin", np.float32).argmax())

    # ---- per-layer records for the fused linear layer (lin_layer/layout.py), from the files above:
    # consts = [lnw | glue side minus its xn slot | nw | postln]; hdr = the MoE header with sgw preloaded
    for l in range(NL):
        hdr = np.zeros(LL.H_BYTES, np.uint8)
        hdr[LL.H_SGW:LL.H_SGW + 4096] = np.fromfile(WDIR / f"sgw{l}.bin", np.uint8)[:4096]
        wr(f"hdr{l}.bin", hdr)
        if not full[l]:
            consts = np.zeros(LL.C_BYTES, np.uint8)
            consts[LL.C_LNW:LL.C_LNW + 4096] = np.fromfile(WDIR / f"lnw{l}.bin", np.uint8)[:4096]
            consts[LL.C_WA:LL.C_NW] = np.fromfile(WDIR / f"side{l}.bin", np.uint8)[4096:]
            consts[LL.C_NW:LL.C_NW + 4096] = np.fromfile(WDIR / f"nw{l}.bin", np.uint8)[:4096]
            consts[LL.C_POSTLN:LL.C_POSTLN + 4096] = np.fromfile(WDIR / f"postln{l}.bin", np.uint8)[:4096]
            wr(f"consts{l}.bin", consts)

    # ---- config
    X = [("L", "ln", "ln/build"), ("Z", "gz", "gemv_q4/build_z"),
         ("A", "la", "lin_layer/build_a"), ("N", "dn", "deltanet/build"), ("C", "lc", "lin_layer/build_c"),
         ("O", "gout", "gemv_q4/build_out"), ("R", "rt", "router/build"), ("E", "me", "moe_experts/build"),
         ("S", "gsu", "gemv_q4/build_share_up"),
         ("H", "g4kh", "gemv_q4/build_z_hi"), ("I", "g512h", "gemv_q4/build_512_hi"), ("X", "at", "attn/build_pos0"),
         ("K", "lm", "lm_head_q8/build_full")]
    cfg = ["device"]
    for tag, kn, path in X:
        cfg += [f"xclbin {tag} {D}/{path}/final.xclbin", f"kernelx {kn} {tag} {D}/{path}/insts.bin"]
    cfg += [f"buf xres0 8192 {OUT}/xres0.bin", f"buf zero 8192 {OUT}/zero.bin", f"buf normw 4096 {OUT}/normw.bin",
            f"buf lmpool 542113792 {OUT}/lm27.bin",
            "buf xresf 8192", "buf hn 4096", "buf logits 993280",
            f"buf zstate 49152 {OUT}/zstate.bin", f"buf zS 2097152 {OUT}/zS.bin", f"buf zkv 3145728 {OUT}/zkv.bin"]
    runs = []
    for l in range(NL):
        cfg += [f"buf rw{l} 1048576 {OUT}/rw{l}.bin", f"buf rout{l} 4096", f"buf xc{l} 8192",
                f"buf pool{l} 536870912 {POOLS}/pool_L{l}.bin", f"buf hdr{l} 20480 {OUT}/hdr{l}.bin"]
        xin = "xres0" if l == 0 else f"xc{l - 1}"
        if not full[l]:
            # fused linear layer: lin_a (ln -> qkv|z -> glue) / dn / lin_c (post -> out -> ln+res),
            # the conv state per layer updated in place, lin_c writing the MoE header
            cfg += [f"buf consts{l} {LL.C_BYTES} {OUT}/consts{l}.bin", f"buf state{l} 49152 {OUT}/zstate.bin",
                    f"buf act{l} {LL.A_BYTES}", f"buf vec{l} 65536", f"buf sout{l} 2097152", f"buf o{l} 16384",
                    f"buf wout{l} 10485760 {OUT}/wout{l}.bin"]
            runs += [f"run la pool{l} {xin} consts{l} state{l} act{l} vec{l}",
                     f"run dn zS vec{l} sout{l} o{l}",
                     f"run lc wout{l} o{l} consts{l} act{l} {xin} hdr{l}"]
        else:
            cfg += [f"buf lnw{l} 4096 {OUT}/lnw{l}.bin", f"buf postln{l} 4096 {OUT}/postln{l}.bin",
                    f"buf xa{l} 8192", f"buf xn{l} 4096", f"buf out{l} 8192", f"buf xb{l} 8192", f"buf xm{l} 4096",
                    f"buf wq{l} 5242880 {OUT}/wq{l}.bin", f"buf wgate{l} 5242880 {OUT}/wgate{l}.bin",
                    f"buf wk{l} 655360 {OUT}/wk{l}.bin", f"buf wv{l} 655360 {OUT}/wv{l}.bin", f"buf wo{l} 5242880 {OUT}/wo{l}.bin",
                    f"buf qg{l} 32768", f"buf kvn{l} 4096", f"buf meta{l} 2048 {OUT}/meta{l}.bin",
                    f"buf kvnew{l} 2048", f"buf og{l} 8192"]
            runs += [f"run ln {xin} zero lnw{l} xa{l} xn{l}",
                     f"run gz wq{l} xn{l} qg{l}", f"run g4kh wgate{l} xn{l} qg{l}",
                     f"run gsu wk{l} xn{l} kvn{l}", f"run g512h wv{l} xn{l} kvn{l}",
                     f"run at meta{l} qg{l} kvn{l} zkv kvnew{l} og{l}",
                     f"run gout wo{l} og{l} out{l}",
                     f"run ln xa{l} out{l} postln{l} xb{l} xm{l}",
                     # header [xm | rout | sgw | xres]: host copies until the fused attention layer (A') writes it
                     f"copy hdr{l} 0 xm{l} 0 4096", f"copy hdr{l} 12288 xb{l} 0 8192"]
        # router (reads xm at hdr+0) -> the whole MoE block as one dispatch
        runs += [f"run rt hdr{l} rw{l} rout{l}", f"moeroute me rout{l}",
                 f"copy hdr{l} 4096 rout{l} 0 4096",
                 f"run me pool{l} hdr{l} xc{l}",
                 f"dump rout{l} {OUT}/y_rout{l}.bin 4096", f"dump xc{l} {OUT}/y_res{l}.bin 8192"]
    runs += [f"run ln xc{NL - 1} zero normw xresf hn", "run lm lmpool hn logits", f"dump logits {OUT}/y_logits.bin 993280", ""]
    (HERE / "run_27b.cfg").write_text("\n".join(cfg + runs), newline="\n")
    print(f"{NL} layers, {len([r for r in runs if r.startswith('run ')])} runs; ref argmax {ref_argmax}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
