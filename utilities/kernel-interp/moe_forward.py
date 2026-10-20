"""Full layer-0 MoE forward from POOL weights; verify against captured L1 input.

Dequant layouts (verified):
  expert gate_up: per expert 8 alternating 163840B bands [up_k | gate_k] k=0..3,
    each band = 32 chunks covering 128 rows x 2048 cols:
      rows = 32*(c%4), cols = 256*(c//4)
  expert down: contiguous 128 chunks per expert, [2048,512]:
      rows = 128*(c//8) + 32*(c%4), cols = 256*((c//4)%2)
"""
import numpy as np
from q4nx import bf16_to_f32, dequant_q4nx_packed, Q4NX, MODEL_DIR
import os

CAP = "C:/caps/m0d"
S = 163840
DOWN0 = 335544320

def dq_chunks(chunks):
    """[n,5120] uint8 -> values [n, 32, 8, 32] f32 (row-tile, blockcol, elem)."""
    nch = chunks.shape[0]
    meta = bf16_to_f32(np.ascontiguousarray(chunks[:, :1024]).view(np.uint16))
    d, m = meta[:, :256], meta[:, 256:]
    q = chunks[:, 1024:]
    n = np.empty((nch, 8192), dtype=np.float32)
    n[:, 0::2] = q & 0xF
    n[:, 1::2] = q >> 4
    r = np.arange(32)[:, None, None]
    bc = np.arange(8)[None, :, None]
    i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16)
    j = (bc * 32 + r + 0 * i)
    vals = n[:, p.reshape(-1)].reshape(nch, 32, 8, 32)
    dd = d[:, j.reshape(-1)].reshape(nch, 32, 8, 32)
    mm_ = m[:, j.reshape(-1)].reshape(nch, 32, 8, 32)
    return vals * dd + mm_

def dequant_band(band_bytes):
    """163840B band -> [128, 2048]"""
    w = dq_chunks(band_bytes.reshape(32, 5120)).reshape(32, 32, 256)
    W = np.empty((128, 2048), dtype=np.float32)
    for c in range(32):
        W[32*(c % 4):32*(c % 4)+32, 256*(c//4):256*(c//4)+256] = w[c]
    return W

def expert_gate_up(pool, e):
    up = np.empty((512, 2048), dtype=np.float32)
    gt = np.empty((512, 2048), dtype=np.float32)
    for k in range(4):
        off_u = (8*e + 2*k) * S
        off_g = (8*e + 2*k + 1) * S
        up[128*k:128*k+128] = dequant_band(pool[off_u:off_u+S])
        gt[128*k:128*k+128] = dequant_band(pool[off_g:off_g+S])
    return gt, up

def expert_down(pool, e):
    chunks = pool[DOWN0 + e*655360: DOWN0 + (e+1)*655360].reshape(128, 5120)
    w = dq_chunks(chunks).reshape(128, 32, 256)
    W = np.empty((2048, 512), dtype=np.float32)
    for c in range(128):
        r0 = 128*(c//8) + 32*(c % 4)
        c0 = 256*((c//4) % 2)
        W[r0:r0+32, c0:c0+256] = w[c]
    return W

def silu(x):
    return x / (1 + np.exp(-x))

if __name__ == "__main__":
    import json
    pool = np.fromfile(f"{CAP}/blob_536870912_836fd8e49f35a0b6.bin", dtype=np.uint8)
    xm = np.load("xm_layer0.npy")
    x_res = np.load("x_res_layer0.npy")
    routing = json.load(open("routing_layer0.json"))

    # sanity: expert 7 vs HF
    gt, up = expert_gate_up(pool, 7)
    GU = np.load("hf_ref/l0_expert7_gate_up_proj.npy")
    print("e7 gate maxerr", np.abs(gt - GU[:512]).max(), "up maxerr", np.abs(up - GU[512:]).max())
    D = expert_down(pool, 7)
    Dref = np.load("hf_ref/l0_expert7_down_proj.npy")
    print("e7 down maxerr", np.abs(D - Dref).max())

    moe = np.zeros((11, 2048), dtype=np.float64)
    for e_str, lst in routing.items():
        e = int(e_str)
        gt, up = expert_gate_up(pool, e)
        D = expert_down(pool, e)
        for t, w in lst:
            h = silu(gt @ xm[t]) * (up @ xm[t])
            moe[t] += w * (D @ h)
    # shared expert from pool
    m = Q4NX(os.path.join(MODEL_DIR, "model_3LiF.q4nx"))
    SGU0, SG0, SD0 = 503316480, 503971840, 504627200
    Wsu = dequant_q4nx_packed(pool[SGU0:SGU0+655360], 512, 2048)
    Wsg = dequant_q4nx_packed(pool[SG0:SG0+655360], 512, 2048)
    # shared down: [2048,512] like expert down? try expert-down layout on its bytes
    sd_chunks = pool[SD0:SD0+655360].reshape(128, 5120)
    w = dq_chunks(sd_chunks).reshape(128, 32, 256)
    Wsd = np.empty((2048, 512), dtype=np.float32)
    for c in range(128):
        r0 = 128*(c//8) + 32*(c % 4); c0 = 256*((c//4) % 2)
        Wsd[r0:r0+32, c0:c0+256] = w[c]
    sgw = m.bf16("model.layer.0.shared_expert_gate.weight")
    sh = silu(xm @ Wsg.T) * (xm @ Wsu.T) @ Wsd.T
    sgate = 1/(1+np.exp(-(xm @ sgw)))
    moe_full = moe + sgate[:, None]*sh
    x1 = x_res + moe_full
    ln1 = m.bf16("model.layer.1.input_layernorm.weight")
    pred = (x1 / np.sqrt((x1**2).mean(-1, keepdims=True)+1e-6) * ln1).astype(np.float32)
    act1 = bf16_to_f32(np.fromfile(f"{CAP}/blob_2097152_70b7719b1b984d1f.bin", dtype=np.uint16)).reshape(512, 2048)[:11]
    d = np.abs(pred - act1)
    print("L1 input match: max", d.max(), "median rel", np.median(d/(np.abs(act1)+1e-3)))
    for t in (0, 5, 10):
        print(f"  token {t} corr {np.corrcoef(pred[t], act1[t])[0,1]:.6f}")
