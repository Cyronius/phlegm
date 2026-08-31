"""Full 30-layer interval-3 model forward on CPU from model_30L.q4nx ONLY.

Generalizes full_forward.py (3LiF) to an arbitrary N-layer schedule derived
from the q4nx tensor names (full_attention where self_attn.q_proj.weight is
present, else linear_attention). Reuses full_forward's per-op math verbatim by
retargeting its module globals (m, T) — every layout/op element was verified
byte-exact vs captured NPU buffers in the 3LiF/5Li3 work.

Purpose: the CPU validation oracle for the 30-layer NPU run. FLM's closed engine
NaN-collapses on interval-3; this open host produces FINITE, healthy logits.

Usage: python l30_forward.py [model.q4nx] [out_dir]
Writes <out>/cpu_prefill_hidden.npy, <out>/cpu_logits.npy, <out>/first_token.npy.
"""
import numpy as np, os, sys
from q4nx import Q4NX, MODEL_DIR, bf16_to_f32
import full_forward as F

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(MODEL_DIR, "model_30L.q4nx")
OUT = sys.argv[2] if len(sys.argv) > 2 else "C:/code/FastFlowLM/npu-engine/m3out/l30"
os.makedirs(OUT, exist_ok=True)

m = Q4NX(MODEL)
F.m = m                                    # retarget full_forward's dequant/layer helpers
ids = np.load("prompt_token_ids.npy")
T = len(ids)
F.T = T

nlayers = 0
while f"model.layer.{nlayers}.input_layernorm.weight" in m.tensors:
    nlayers += 1
sched = ["full_attention" if f"model.layer.{l}.self_attn.q_proj.weight" in m.tensors
         else "linear_attention" for l in range(nlayers)]
print(f"model: {os.path.basename(MODEL)}  nlayers={nlayers}")
print("schedule:", "".join("F" if s == "full_attention" else "L" for s in sched))


def rms(x, eps=1e-6):
    return x / np.sqrt((np.asarray(x, dtype=np.float64) ** 2).mean(-1, keepdims=True) + eps)


def lmhead_logits(hn):
    """q8 lm_head (FILE raster order) -> [248320] fp32. Copied from full_forward."""
    lmb = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
    d = bf16_to_f32(np.ascontiguousarray(lmb[:, :512]).view(np.uint16))
    qq = np.ascontiguousarray(lmb[:, 512:]).view(np.int8)
    nch = lmb.shape[0]
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
    j = bc * 32 + r + 0 * i
    logits = np.zeros(248320, dtype=np.float32)
    CH = 2048
    for c0 in range(0, nch, CH):
        ce = min(c0 + CH, nch)
        vals = qq[c0:ce][:, p.reshape(-1)].reshape(ce - c0, 32, 8, 32).astype(np.float32)
        dd = d[c0:ce][:, j.reshape(-1)].reshape(ce - c0, 32, 8, 32)
        w = (vals * dd).reshape(ce - c0, 32, 256)
        for cc in range(c0, ce):
            row0 = 32 * (cc // 8); col0 = 256 * (cc % 8)
            logits[row0:row0 + 32] += w[cc - c0] @ hn[col0:col0 + 256]
    return logits


if __name__ == "__main__":
    t0 = m.tensors["model.embed_tokens.weight"]
    base = m.data_base + t0["data_offsets"][0]
    E = np.stack([bf16_to_f32(np.frombuffer(m.mm[base + i * 4096: base + (i + 1) * 4096], dtype=np.uint16)) for i in ids])
    x = E.astype(np.float64)
    fpos = np.arange(T).astype(np.float64)

    for l, lt in enumerate(sched):
        if lt == "linear_attention":
            x = F.moe_block(l, F.linear_attn_layer(l, x))
        else:
            x = F.moe_block(l, F.full_attn_layer(l, x, fpos))
        am = float(np.abs(x).max())
        finite = bool(np.isfinite(x).all())
        print(f"  L{l:2d} ({'F' if lt=='full_attention' else 'L'}) absmax={am:10.4f} finite={finite}")
        if not finite:
            print("  !! NON-FINITE — interval-3 blowup on CPU (should NOT happen in open host)")

    hn = (rms(x[-1]) * m.bf16("model.norm.weight")).astype(np.float32)
    np.save(f"{OUT}/cpu_prefill_hidden.npy", hn)
    logits = lmhead_logits(hn)
    np.save(f"{OUT}/cpu_logits.npy", logits)
    first = int(logits.argmax())
    np.save(f"{OUT}/first_token.npy", np.array(first))
    print(f"\nlogits: finite={bool(np.isfinite(logits).all())} absmax={float(np.abs(logits).max()):.3f}")
    print(f"argmax(first token)={first}  top5={np.argsort(logits)[-5:][::-1].tolist()}")
    print(f"residual-stream absmax (last layer)={float(np.abs(x[-1]).max()):.3f}")
    print("DONE — wrote cpu_prefill_hidden.npy / cpu_logits.npy / first_token.npy to", OUT)
