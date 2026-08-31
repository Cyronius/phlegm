"""Build NPU weight pools from the q4nx file (the loader-permutation step).

Pool chunk laws (all verified against captured pools / HF weights):
  standard matmul tensor [out, in]: pool chunk c covers
     rows0 = 64*(c//per_band) + 32*(c%2), per_band = in//128
     cols0 = 1024*((c//8) % max(1,in//1024)) + 256*((c//2)%4)
  file raster: chunk f covers rows0 = 32*(f//ncol), cols0 = 256*(f%ncol)
  -> permute file chunks into pool order by matching (rows0, cols0).
  expert gate_up: pool = interleaved 163840B stripes [up_k | gate_k] x4/expert,
     within-stripe transpose pool_c = 4*(f%8) + (f//8)
  expert/shared down [2048,512]: pool_c = 8*(rt//4) + 4*cg + rt%4  (f = 2rt+cg)
Layer-pool template (from L0/L1 captures):
  [0)          gate_up expert stripes
  [335544320)  down experts (permuted per expert)
  [503316480)  share_up   [503971840) share_gate   [504627200) share_down
  [505282560)  main proj A (linear: qkv / full-attn: q_proj)   10485760 B
  [515768320)  main proj B (linear: z-gate / full-attn: o_proj) 5242880 B
  [521011200)  (full-attn: k_proj 655360) [521666560) (v_proj 655360)  -- guesses
lm_head pool: q8 chunks (8704B) in standard-in2048 pool order at 0; tail zeros.
"""
import numpy as np, os, sys
from q4nx import Q4NX, MODEL_DIR

CH = 5120

def std_perm(nch, out_dim, in_dim, chunk_bytes=CH):
    """pool chunk index -> file chunk index (standard law <-> raster)"""
    ncol = in_dim // 256
    per_band = in_dim // 128
    kgroups = max(1, in_dim // 1024)
    perm = np.zeros(nch, dtype=np.int64)
    for c in range(nch):
        rows0 = 64*(c//per_band) + 32*(c % 2)
        cols0 = (1024*((c//8) % kgroups) + 256*((c//2) % 4)) % in_dim
        f = (rows0//32)*ncol + cols0//256
        perm[c] = f
    return perm

def permute_chunks(raw, perm, chunk_bytes=CH):
    src = np.frombuffer(raw, dtype=np.uint8).reshape(-1, chunk_bytes)
    return src[perm].reshape(-1)

def down_perm():
    perm = np.zeros(128, dtype=np.int64)
    for c in range(128):
        rt = 4*(c//8) + (c % 4)
        cg = (c//4) % 2
        perm[c] = 2*rt + cg
    return perm

def stripe_transpose():
    perm = np.zeros(32, dtype=np.int64)
    for c in range(32):
        rt = c % 4
        cg = c // 4
        perm[c] = 8*rt + cg
    return perm

def build_layer_pool(m, layer, full_attn):
    S = 163840
    pool = np.zeros(536870912, dtype=np.uint8)
    up = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.up_exps_proj.weight"), dtype=np.uint8)
    gt = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.gate_exps_proj.weight"), dtype=np.uint8)
    dn = np.frombuffer(m.raw(f"model.layer.{layer}.mlp.down_exps_proj.weight"), dtype=np.uint8)
    tp = stripe_transpose()
    for e in range(256):
        for k in range(4):
            us = up[(4*e+k)*S:(4*e+k+1)*S].reshape(32, CH)[tp].reshape(-1)
            gs = gt[(4*e+k)*S:(4*e+k+1)*S].reshape(32, CH)[tp].reshape(-1)
            pool[(8*e+2*k)*S:(8*e+2*k+1)*S] = us
            pool[(8*e+2*k+1)*S:(8*e+2*k+2)*S] = gs
    dp = down_perm()
    for e in range(256):
        seg = dn[e*655360:(e+1)*655360].reshape(128, CH)[dp].reshape(-1)
        pool[335544320+e*655360:335544320+(e+1)*655360] = seg
    p128 = std_perm(128, 512, 2048)
    pool[503316480:503316480+655360] = permute_chunks(m.raw(f"model.layer.{layer}.mlp.share_up_exps_proj.weight"), p128)
    pool[503971840:503971840+655360] = permute_chunks(m.raw(f"model.layer.{layer}.mlp.share_gate_exps_proj.weight"), p128)
    pool[504627200:504627200+655360] = permute_chunks(m.raw(f"model.layer.{layer}.mlp.share_down_exps_proj.weight"), std_perm(128, 2048, 512))
    if full_attn:
        # layout decoded from op-ctrlcode addresses (pool device base 0x20000):
        # [q-half 5242880][k 655360][v 655360][gate-half 5242880][o 5242880]
        qg = np.frombuffer(m.raw(f"model.layer.{layer}.self_attn.q_proj.weight"), dtype=np.uint8).reshape(-1, CH)
        p1024 = std_perm(1024, 4096, 2048)
        pool[505282560:505282560+5242880] = qg[:1024][p1024].reshape(-1)          # q half
        pool[510525440:510525440+655360] = permute_chunks(m.raw(f"model.layer.{layer}.self_attn.k_proj.weight"), std_perm(128, 512, 2048))
        pool[511180800:511180800+655360] = permute_chunks(m.raw(f"model.layer.{layer}.self_attn.v_proj.weight"), std_perm(128, 512, 2048))
        pool[511836160:511836160+5242880] = qg[1024:][p1024].reshape(-1)          # gate half
        pool[517079040:517079040+5242880] = permute_chunks(m.raw(f"model.layer.{layer}.self_attn.o_proj.weight"), std_perm(1024, 2048, 4096))
    else:
        pool[505282560:505282560+10485760] = permute_chunks(m.raw(f"model.layer.{layer}.linear_attn.qkv_proj.weight"), std_perm(2048, 8192, 2048))
        pool[515768320:515768320+5242880] = permute_chunks(m.raw(f"model.layer.{layer}.self_attn.gate_proj.weight"), std_perm(1024, 4096, 2048))
    return pool

def build_lmhead_pool(m):
    """lm_head q8 pool: 128-row supertile transpose (decoded via one-hot probes),
    NOT the standard matmul perm. pool chunk k <- file chunk
    (4*(k//32) + (k%4))*8 + ((k%32)//4)."""
    raw = np.frombuffer(m.raw("lm_head.weight"), dtype=np.uint8).reshape(-1, 8704)
    nch = raw.shape[0]
    perm = np.zeros(nch, dtype=np.int64)
    for k in range(nch):
        s, r = divmod(k, 32)
        cg, rg = r // 4, r % 4
        perm[k] = (4 * s + rg) * 8 + cg
    out = np.zeros(542113792, dtype=np.uint8)
    out[:nch * 8704] = raw[perm].reshape(-1)
    return out

if __name__ == "__main__":
    m = Q4NX(os.path.join(MODEL_DIR, "model_3LiF.q4nx"))
    outdir = sys.argv[1] if len(sys.argv) > 1 else "C:/code/FastFlowLM/npu-engine/m3out"
    # sanity: rebuild L0 pool and byte-compare vs captured
    p0 = build_layer_pool(m, 0, False)
    cap = np.fromfile("C:/caps/m0d/blob_536870912_836fd8e49f35a0b6.bin", dtype=np.uint8)
    regions = [("gate_up", 0, 335544320), ("down", 335544320, 503316480),
               ("share_up", 503316480, 503971840), ("share_gate", 503971840, 504627200),
               ("share_down", 504627200, 505282560), ("qkv", 505282560, 515768320),
               ("z", 515768320, 521011200), ("tail", 521011200, 536870912)]
    for nm, a, b in regions:
        eq = np.array_equal(p0[a:b], cap[a:b])
        nd = int((p0[a:b] != cap[a:b]).sum())
        print(f"L0 rebuild {nm:10s}: {'MATCH' if eq else f'{nd} bytes differ'}")
    p2 = build_layer_pool(m, 2, True)
    p2.tofile(f"{outdir}/pool_L2.bin")
    lm = build_lmhead_pool(m)
    lm.tofile(f"{outdir}/pool_lmhead.bin")
    print("wrote pool_L2.bin / pool_lmhead.bin")


# ---- pack (2MB) and side (6MB) builders --------------------------------------
def build_pack(m, layer):
    """[ln@0][postln@4096][sgate@8192][router@12288..1060863] + zeros"""
    pk = np.zeros(2097152, dtype=np.uint8)
    def put(off, name):
        b = np.frombuffer(m.raw(name), dtype=np.uint8)
        pk[off:off+len(b)] = b
    put(0,     f"model.layer.{layer}.input_layernorm.weight")
    put(4096,  f"model.layer.{layer}.post_attention_layernorm.weight")
    put(8192,  f"model.layer.{layer}.shared_expert_gate.weight")
    put(12288, f"model.layer.{layer}.moe_router.weight")
    return pk

def build_side(m, layer, full_attn):
    side = np.zeros(6291456, dtype=np.uint8)
    def put(off, name):
        b = np.frombuffer(m.raw(name), dtype=np.uint8)
        side[off:off+len(b)] = b
    if full_attn:
        put(128, f"model.layer.{layer}.self_attn.q_norm.weight")
        put(640, f"model.layer.{layer}.self_attn.k_norm.weight")
    else:
        put(0,      f"model.layer.{layer}.linear_attn.ssm_conv1d.weight")   # 65536
        put(65536,  f"model.layer.{layer}.linear_attn.ssm_norm.weight")     # 256
        put(65792,  f"model.layer.{layer}.linear_attn.ssm_a")               # 128 (f32)
        put(65920,  f"model.layer.{layer}.linear_attn.ssm_dt.bias")         # 128
        put(66048,  f"model.layer.{layer}.linear_attn.ssm_alpha_proj.weight")  # 131072
        put(197120, f"model.layer.{layer}.linear_attn.ssm_beta_proj.weight")   # 131072
        # out_proj: q4, needs pool permutation (std law, [2048,4096])
        side[328192:328192+5242880] = permute_chunks(
            m.raw(f"model.layer.{layer}.linear_attn.ssm_out_proj.weight"), std_perm(1024, 2048, 4096))
    return side
