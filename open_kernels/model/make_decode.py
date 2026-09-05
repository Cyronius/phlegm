r"""Build everything needed to run N decode steps of a stock Qwen3.6-MoE model
on the open kernels, and the fp64 reference to judge them by.

    python open_kernels/model/make_decode.py --layers 8 --tokens 1
    open_kernels/harness/out/run_kernel.exe open_kernels/model/out/run_decode.cfg
    python open_kernels/model/compare_decode.py --tokens 1

Emitted per layer: the 512 MB weight pool and a `consts` blob holding the
layer's small weights, both packed by the recipe's plan (recipes/pack.py over
recipes/qwen36moe.pack_plan -- the same plan src/open_qwen36/pools.cpp
interprets), and a zeroed state / KV buffer. Emitted once: the lm_head pool,
the RoPE position table, and the reference residuals + logits from replica.py.
The `.cfg` is the manifest's per-layer program spelled out in the harness's
directives.

Decode starts at position 0 from zeroed state, which is exact prefill for this
architecture (each layer's state update sees one token at a time), and each next
token is the reference's greedy pick -- so the NPU and the reference run the same
token sequence and every position is comparable.

The routed experts are chosen ON DEVICE: the layer kernel's first dispatch ends
with the router, the driver's `moeroute2` points the expert fills at what it
picked, and the second dispatch runs the MoE block. The reference re-uses the
NPU's choice when a previous run's `y_rout` file disagrees with its own (the 8th
slot is often a near-tie), so a routing difference doesn't masquerade as a
numerical one -- it is printed when it happens.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DESIGNS = HERE.parent / "designs"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from recipes import pack as PK  # noqa: E402
from recipes import qwen36moe as Q  # noqa: E402
from recipes.load import spec_from_model_dir  # noqa: E402
from recipes.spec import FULL  # noqa: E402
import replica as R  # noqa: E402
from q4nx import Q4NX, f32_to_bf16  # noqa: E402

DEFAULT_MODEL_DIR = os.environ.get(
    "FLM_MODEL_DIR", str(Path.home() / ".flm" / "models" / "Qwen3.6-35B-A3B-NPU2"))   # FLM's default model store


def sfx(t):
    return "" if t == 0 else f"_t{t}"


def write(path: Path, arr):
    path.write_bytes(np.ascontiguousarray(arr).tobytes())


def npu_routing(out: Path, layer, t):
    """The 8 experts the NPU's router picked in a previous run, if there is one."""
    p = out / f"y_rout{layer}{sfx(t)}.bin"
    if not p.is_file():
        return None
    return np.fromfile(p, np.float32)[256:264].view(np.int32).astype(np.int64)


def build_cfg(args, nl, full, out: Path, pool_dir: Path, L: Q.Layout) -> str:
    d = DESIGNS.as_posix()
    o = out.as_posix()
    cfg = ["device"]
    # Only declare the design a slice actually uses: a short slice can be all
    # linear layers, and registering an xclbin costs a hw_context either way.
    if any(not full[l] for l in range(nl)):
        cfg += [f"xclbin X {d}/layer_x/build_lx0/final.xclbin",
                f"kernelx lx0 X {d}/layer_x/build_lx0/insts.bin",
                f"kernelx lx1 X {d}/layer_x/build_lx1/insts.bin"]
    if any(full[l] for l in range(nl)):
        cfg += [f"xclbin Y {d}/layer_x/build_ax0/final.xclbin",
                f"kernelx ax0 Y {d}/layer_x/build_ax0/insts.bin",
                f"kernelx ax1 Y {d}/layer_x/build_ax1/insts.bin"]
    cfg += [f"xclbin L {d}/ln/build/final.xclbin",
            f"kernelx ln L {d}/ln/build/insts.bin",
            f"xclbin K {d}/lm_head_q8/build_full/final.xclbin",
            f"kernelx lm K {d}/lm_head_q8/build_full/insts.bin",
            f"buf xres 8192 {o}/xres0.bin",
            f"buf zero 8192 {o}/zero.bin",
            f"buf normw 4096 {o}/normw.bin",
            f"buf lmpool {L.LMHEAD_POOL_BYTES} {pool_dir.as_posix()}/pool_lmhead.bin",
            "buf xresf 8192", "buf hn 4096", "buf logits 993280",
            f"buf ptab {L.PTAB_BYTES} {o}/ptab.bin"]
    for l in range(nl):
        cfg.append(f"buf pool{l} {L.POOL_BYTES} {pool_dir.as_posix()}/pool_L{l}.bin")
        if full[l]:
            cfg += [f"buf constsa{l} {L.CA_BYTES} {o}/constsa_{l}.bin",
                    f"buf acta{l} {L.AA_BYTES}", f"buf kv{l} {L.KV_BYTES}"]
        else:
            cfg += [f"buf consts{l} {L.C_BYTES} {o}/consts_{l}.bin",
                    f"buf state{l} {L.STATE_BYTES} {o}/zstate.bin",
                    f"buf act{l} {L.A_BYTES}"]
    runs = []
    for t in range(args.tokens):
        s = sfx(t)
        if t:
            runs.append(f"load xres {o}/xres{t}.bin")
        if any(full[l] for l in range(nl)):
            runs.append(f"attnpos ax0 {t}")   # the attention layers share one stream
        for l in range(nl):
            if full[l]:
                runs += [f"run ax0 pool{l} xres constsa{l} kv{l} acta{l} ptab",
                         f"moeroute2 ax1 acta{l} {L.AA_ROUT + Q.ROUT_IDX_OFF}",
                         f"run ax1 pool{l} xres constsa{l} kv{l} acta{l} ptab",
                         f"dump acta{l} {o}/y_rout{l}{s}.bin 4096 {L.AA_ROUT}"]
            else:
                runs += [f"run lx0 pool{l} xres consts{l} state{l} act{l}",
                         f"moeroute2 lx1 act{l} {L.A_ROUT + Q.ROUT_IDX_OFF}",
                         f"run lx1 pool{l} xres consts{l} state{l} act{l}",
                         f"dump act{l} {o}/y_rout{l}{s}.bin 4096 {L.A_ROUT}"]
            runs.append(f"dump xres {o}/y_res{l}{s}.bin 8192")
        runs += ["run ln xres zero normw xresf hn",
                 "run lm lmpool hn logits",
                 f"dump logits {o}/y_logits{s}.bin 993280"]
    return "\n".join(cfg + runs) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--layers", type=int, default=8,
                    help="run the first N layers (each holds a 512 MB pool on the device)")
    ap.add_argument("--tokens", type=int, default=1, help="decode N tokens, positions 0..N-1")
    ap.add_argument("--token", type=int, default=248045, help="the first token id")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--pool-dir", default=None, help="where the big pools live (default <out>/pools)")
    ap.add_argument("--reuse-pools", action="store_true", help="keep pool files that already exist")
    ap.add_argument("--cfg-only", action="store_true", help="only rewrite the .cfg")
    ap.add_argument("--strict-routing", action="store_true",
                    help="keep the reference's own top-8 even where a previous run shows the NPU "
                         "picked differently (by default the NPU's choice is adopted, so an 8th-slot "
                         "near-tie doesn't read as a numerical error)")
    a = ap.parse_args()

    # Absolute: the .cfg resolves relative paths against its own directory.
    out = Path(a.out).resolve()
    pool_dir = Path(a.pool_dir).resolve() if a.pool_dir else out / "pools"
    out.mkdir(parents=True, exist_ok=True)
    pool_dir.mkdir(parents=True, exist_ok=True)

    md = Path(a.model_dir)
    spec = spec_from_model_dir(md)
    L = Q.layout(spec)
    plan = Q.pack_plan(spec)
    nl = min(a.layers, spec.num_layers)
    full = {l: spec.layer_types[l] == FULL for l in range(nl)}
    m = Q4NX(md / "model.q4nx")
    print(f"{md.name}: {spec.num_layers} layers -> running {nl}, attention at {[l for l in range(nl) if full[l]]}")

    if not a.cfg_only:
        write(out / "zero.bin", np.zeros(spec.hidden, np.float32))
        write(out / "zstate.bin", np.zeros(L.STATE_BYTES, np.uint8))
        write(out / "ptab.bin", PK.ptab(L.MAX_CTX, spec.rotary_dim, spec.rope_theta, L.PTAB_ROW))
        write(out / "normw.bin", f32_to_bf16(m.bf16(plan["norm"]["tensor"])))

        for l in range(nl):
            lt = spec.layer_types[l]
            pf = pool_dir / f"pool_L{l}.bin"
            if not (a.reuse_pools and pf.is_file() and pf.stat().st_size == L.POOL_BYTES):
                PK.build_layer_pool(plan, lt, m, l).tofile(pf)
            if full[l]:
                write(out / f"constsa_{l}.bin", PK.build_consts(plan, lt, m, l, L.CA_BYTES))
            else:
                write(out / f"consts_{l}.bin", PK.build_consts(plan, lt, m, l, L.C_BYTES))
            print(f"  layer {l} {'FULL' if full[l] else 'lin '}: pool + consts", flush=True)
        lmf = pool_dir / "pool_lmhead.bin"
        if not (a.reuse_pools and lmf.is_file() and lmf.stat().st_size == L.LMHEAD_POOL_BYTES):
            PK.build_lmhead_pool(plan, m).tofile(lmf)
            print("  lm_head pool", flush=True)

        # ---- the reference: the same token sequence, in fp64, on CPU
        conv = {l: np.zeros((spec.conv_kernel - 1, spec.lin_qkv_dim)) for l in range(nl)}
        S = {l: np.zeros((spec.lin_value_heads, spec.lin_value_dim, spec.lin_value_dim)) for l in range(nl)}
        K = {l: np.zeros((0, spec.num_kv_heads, spec.head_dim)) for l in range(nl)}
        V = {l: np.zeros((0, spec.num_kv_heads, spec.head_dim)) for l in range(nl)}
        tok = a.token
        for t in range(a.tokens):
            x = m.embed(tok)
            write(out / f"xres{t}.bin", x.astype(np.float32))
            xr = x.copy()
            for l in range(nl):
                if full[l]:
                    xa, K[l], V[l] = R.attn_decode(m, l, xr.copy(), K[l], V[l], t)
                else:
                    xa, conv[l], S[l] = R.linear_decode(m, l, xr.copy(), conv[l], S[l])
                _, _, mine = R.route(m, l, xa)
                got = None if a.strict_routing else npu_routing(out, l, t)
                if got is not None and sorted(got.tolist()) != sorted(mine.tolist()):
                    print(f"    token {t} layer {l}: NPU routed {sorted(got.tolist())}, "
                          f"reference {sorted(mine.tolist())} -- using the NPU's")
                    mine = got
                xr = R.moe_decode(m, l, xa, top=mine)
                write(out / f"ref_res{l}{sfx(t)}.bin", xr.astype(np.float32))
                print(f"  token {t} layer {l} {'FULL' if full[l] else 'lin '} top8={mine.tolist()}", flush=True)
            _, logits = R.final_logits(m, xr)
            write(out / f"ref_logits{sfx(t)}.bin", logits.astype(np.float32))
            tok = int(logits.argmax())
            print(f"  token {t} (position {t}): reference argmax {tok}", flush=True)

    cfg = build_cfg(a, nl, full, out, pool_dir, L)
    (out / "run_decode.cfg").write_text(cfg, newline="\n")
    nruns = len([r for r in cfg.splitlines() if r.startswith("run ")])
    print(f"wrote {out / 'run_decode.cfg'}: {nl} layers, {a.tokens} token(s), {nruns} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
