"""Transcode a verified FLM 1.0.2 (q4_1) q4nx file into a genuine 1.0.3 (Q4_K)
file, using the REFERENCE converter's real packer + the real 1.0.3 reorders.

This is the integration ground truth: running full_forward on the 1.0.2 file and
on this transcoded 1.0.3 file must give matching logits (up to the extra q4_1->
Q4_K requant), which exercises the whole 1.0.3 READ path end-to-end -- Q4_K
dequant, q8 column-major lm_head, and every linear-attention reorder-undo.

  python transcode_102_to_103.py <in_1.0.2.q4nx> <out_1.0.3.q4nx>
"""
import os, sys, importlib.util
import numpy as np
import torch
from einops import rearrange

HERE = os.path.dirname(os.path.abspath(__file__))
KI = os.path.join(HERE, "..", "kernel-interp")
sys.path.insert(0, KI)
import q4nx_v103 as V                                 # unique name, safe

# Reference converter package owns the name `q4nx`; load it for the packer.
sys.path.insert(0, os.path.join(HERE, "reference"))
from q4nx import model_converter as mc
BASE = next(v for v in vars(mc).values()
            if isinstance(v, type) and hasattr(v, "_pack_q4k") and hasattr(v, "_pack_q8nx"))

# Load kernel-interp's fmt-aware reader under a DISTINCT module name to avoid
# clashing with the reference `q4nx` package.
_spec = importlib.util.spec_from_file_location("ki_q4nx", os.path.join(KI, "q4nx.py"))
_kq = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_kq)
Q4NX = _kq.Q4NX

G, P = V.N_HEADS, V.STATE_SIZE


class Shim:
    row_block_size = 32; col_block_size = 256; parallel_size = 32; keep_block_in_2D = True


SHIM = Shim()


# ---- forward 1.0.3 reorders (inverse of q4nx_v103.apply_undo) -----------------
def fwd_qgp_rows(W):   return rearrange(W, '(q g p) c -> (g q p) c', q=2, g=G, p=P)
def fwd_qgp_cols(W):   return rearrange(W, 'r (q g p) -> r (g q p)', q=2, g=G, p=P)
def fwd_qg_cols(W):    return rearrange(W, 'r (q g) -> r (g q)', q=2, g=G)
def fwd_qg_vec(v):     return rearrange(v, '(q g) -> (g q)', q=2, g=G)


def make_tuq(W):
    """fp [out,in] -> (t,u,q) with w = t*q - u exactly (per row, 32-col group)."""
    out, in_ = W.shape
    g = W.reshape(out, in_ // 32, 32)
    lo = g.min(-1); hi = g.max(-1)
    scale = np.where(hi > lo, (hi - lo) / 15.0, 1.0).astype(np.float32)
    q = np.clip(np.rint((g - lo[..., None]) / scale[..., None]), 0, 15).astype(np.float32)
    return (torch.from_numpy(scale), torch.from_numpy((-lo).astype(np.float32)),
            torch.from_numpy(q.reshape(out, in_)))


def pack_q4k(Wfp):
    t, u, q = make_tuq(np.ascontiguousarray(Wfp, np.float32))
    return BASE._pack_q4k(SHIM, t, u, q).view(torch.int8)     # [p,u,4736]


def pack_q8(Wfp):
    out, in_ = Wfp.shape
    g = Wfp.reshape(out, in_ // 32, 32)
    gmax = np.abs(g).max(-1)
    scale = np.where(gmax > 0, gmax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(g / scale[..., None]), -127, 127).reshape(out, in_).astype(np.int8)
    return BASE._pack_q8nx(SHIM, torch.from_numpy(scale.astype(np.float16)), None,
                           torch.from_numpy(q)).view(torch.int8)


def transcode(inp, outp):
    m = Q4NX(inp)
    assert m.fmt == "q4_1", "expected a 1.0.2 source"
    out_tensors = {}
    for name, meta in m.tensors.items():
        dt, shape = meta["dtype"], meta["shape"]
        lt = m._layer_type(name)
        if dt == "F32":                                   # ssm_a / ssm_dt.bias
            v = np.array(m.f32(name), np.float32)
            if lt == "linear" and (name.endswith("ssm_a") or "ssm_dt.bias" in name):
                v = fwd_qg_vec(v)
            out_tensors[name] = torch.from_numpy(v).contiguous()
        elif dt == "BF16":
            w = np.array(m.bf16(name), np.float32)         # logical order
            if lt == "linear" and ("ssm_alpha_proj" in name or "ssm_beta_proj" in name):
                w = fwd_qg_cols(w)                          # [2048,32]
            elif lt == "linear" and "ssm_conv1d" in name:  # [4,8192], v-half cols
                w = w.copy(); w[:, 4096:8192] = fwd_qgp_cols(w[:, 4096:8192])
            out_tensors[name] = torch.from_numpy(np.ascontiguousarray(w)).to(torch.bfloat16)
        elif dt == "I8":
            out_dim, in_dim = shape[0] * 32, shape[1] * 256
            if name == "lm_head.weight":
                out_tensors[name] = pack_q8(_dq_lmhead(m))
            elif "exps_proj" in name and "share" not in name:   # 256 experts stacked
                ne = 256
                oe = out_dim // ne
                raw = np.frombuffer(m.raw(name), np.uint8)
                stride = 128 * m.chunk_bytes
                blocks = []
                for e in range(ne):
                    We = m.dq_tile(raw[e * stride:(e + 1) * stride], oe, in_dim)  # no reorder
                    blocks.append(pack_q4k(We))
                out_tensors[name] = torch.cat(blocks, dim=0).contiguous()
            else:
                W = m.dq_tile(m.raw(name), out_dim, in_dim)     # logical (q_proj already planar)
                if lt == "linear" and "linear_attn.qkv_proj" in name:
                    W = W.copy(); W[4096:8192] = fwd_qgp_rows(W[4096:8192])
                elif lt == "linear" and "self_attn.gate_proj" in name:
                    W = fwd_qgp_rows(W)
                elif lt == "linear" and "ssm_out_proj" in name:
                    W = fwd_qgp_cols(W)
                out_tensors[name] = pack_q4k(W)
        print(f"  {name:60s} {dt} -> {tuple(out_tensors[name].shape)}")
    from safetensors.torch import save_file
    save_file(out_tensors, outp)
    print("wrote", outp)


def _dq_lmhead(m):
    """dequant the 1.0.2 (16-lane q8) lm_head to fp [vocab, hidden]."""
    lm = np.frombuffer(m.raw("lm_head.weight"), np.uint8).reshape(-1, 8704)
    nch = lm.shape[0]
    d = V.bf16_to_f32(np.ascontiguousarray(lm[:, :512]).view(np.uint16))
    qq = np.ascontiguousarray(lm[:, 512:]).view(np.int8)
    r = np.arange(32)[:, None, None]; bc = np.arange(8)[None, :, None]; i = np.arange(32)[None, None, :]
    p = (r // 16) * 4096 + bc * 512 + i * 16 + (r % 16); j = bc * 32 + r + 0 * i
    W = np.empty((nch // 8 * 32, 2048), np.float32)   # [vocab, hidden]
    for cc in range(nch):
        vals = qq[cc][p.reshape(-1)].reshape(32, 8, 32).astype(np.float32)
        dd = d[cc][j.reshape(-1)].reshape(32, 8, 32)
        w = (vals * dd).reshape(32, 256)
        W[32 * (cc // 8):32 * (cc // 8) + 32, 256 * (cc % 8):256 * (cc % 8) + 256] = w
    return W


if __name__ == "__main__":
    transcode(sys.argv[1], sys.argv[2])
