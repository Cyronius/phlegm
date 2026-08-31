"""Validate the open engine's FLM 1.0.3 (Q4_K) reader against the REFERENCE
converter's own packer (FastFlowLM/FLM_Q4NX_Converter, vendored under reference/).

gguf 0.19 cannot ENCODE Q4_K, so we can't route a synthetic GGUF through the
reference converter to get 1.0.3 output.  Instead we drive the reference packer
directly: build (t,u,q) triples from known fp weights, run them through the real
`_pack_q4k` / `_pack_q8nx` / `_refit_one_side`, then read the bytes back with our
reader (tools/kernel-interp/q4nx_v103.py).  Correct layout+refit inversion ->
error at the refit floor (~1e-3 rel); a wrong layout or reorder -> ~O(weight).

  Test A: Q4_K chunk layout + refit round-trip.
  Test B: Q8_0 lm_head column-major layout.
  Test C: each linear-attention reorder-undo.
"""
import os, sys
import numpy as np
import torch
from einops import rearrange

HERE = os.path.dirname(os.path.abspath(__file__))
# import q4nx_v103 (unique name) from kernel-interp first...
sys.path.insert(0, os.path.join(HERE, "..", "kernel-interp"))
import q4nx_v103 as R
# ...then make the reference `q4nx` PACKAGE win over kernel-interp's q4nx.py module.
sys.path.insert(0, os.path.join(HERE, "reference"))
from q4nx import model_converter as mc
from q4nx.gguf_tensor import _refit_one_side  # noqa: F401  (used indirectly by _pack_q4k)

# The abstract base carries _pack_q4k / _pack_q8nx; grab it by capability.
BASE = next(v for v in vars(mc).values()
            if isinstance(v, type) and hasattr(v, "_pack_q4k") and hasattr(v, "_pack_q8nx"))


class Shim:
    row_block_size = 32
    col_block_size = 256
    parallel_size = 32
    keep_block_in_2D = True


SHIM = Shim()
rng = np.random.default_rng(0)


def make_tuq(W):
    """fp [out,in] -> (t,u,q) with w = t*q - u exactly, per (row, 32-col group)."""
    out, in_ = W.shape
    g = W.reshape(out, in_ // 32, 32)
    lo = g.min(-1); hi = g.max(-1)
    scale = np.where(hi > lo, (hi - lo) / 15.0, 1.0).astype(np.float32)
    q = np.clip(np.rint((g - lo[..., None]) / scale[..., None]), 0, 15).astype(np.float32)
    t = scale                                   # (out, in//32)
    u = (-lo).astype(np.float32)                # w = t*q + lo = t*q - u
    return (torch.from_numpy(t), torch.from_numpy(u),
            torch.from_numpy(q.reshape(out, in_)))


def recon_tuq(t, u, q):
    """the exact weights (t*q - u) the (t,u,q) encode, for comparison."""
    t = t.numpy(); u = u.numpy(); q = q.numpy()
    out, ng = t.shape
    qg = q.reshape(out, ng, 32)
    return (t[..., None] * qg - u[..., None]).reshape(out, out and q.shape[1])


def pack_q4k(W):
    t, u, q = make_tuq(W)
    packed = BASE._pack_q4k(SHIM, t, u, q).view(torch.uint8).numpy()  # [p,u,4736]
    return np.ascontiguousarray(packed), recon_tuq(t, u, q)


def relerr(a, b):
    d = np.abs(a - b)
    return float(d.max()), float(d.mean() / (np.abs(b).mean() + 1e-9))


def test_A():
    print("== Test A: Q4_K chunk layout + refit ==")
    for (out, in_) in [(32, 256), (64, 512), (256, 2048), (2048, 512)]:
        W = (rng.standard_normal((out, in_)) * 0.05).astype(np.float32)
        packed, recon = pack_q4k(W)
        got = R.dequant_q4k_file(packed, out, in_)
        mx, rel = relerr(got, recon)
        ok = rel < 5e-3
        print(f"  [{out:4d}x{in_:4d}] maxabs {mx:.4e} rel {rel:.4e}  {'OK' if ok else 'FAIL'}")
        assert ok, "layout wrong (>> refit floor)"


def test_B():
    print("== Test B: Q8_0 lm_head column-major ==")
    for (out, in_) in [(32, 256), (256, 2048)]:
        W = (rng.standard_normal((out, in_)) * 0.05).astype(np.float32)
        gmax = np.abs(W.reshape(out, in_ // 32, 32)).max(-1)
        scale = np.where(gmax > 0, gmax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.rint(W.reshape(out, in_ // 32, 32) / scale[..., None]), -127, 127)
        d = torch.from_numpy(scale.astype(np.float16))                 # _pack_q8nx wants f16
        qw = torch.from_numpy(q.reshape(out, in_).astype(np.int8))
        packed = BASE._pack_q8nx(SHIM, d, None, qw).view(torch.uint8).numpy()
        packed = np.ascontiguousarray(packed)
        # ground truth = bf16(scale) * q  (bf16 because the packer stores bf16 scales)
        sb = R.bf16_to_f32(((scale.view(np.uint32) + 0x8000) >> 16).astype(np.uint16))
        recon = (sb[..., None] * q).reshape(out, in_)
        got = R.dequant_q8_q4k_file(packed, out, in_)
        mx, rel = relerr(got, recon)
        ok = rel < 5e-3
        print(f"  [{out:4d}x{in_:4d}] maxabs {mx:.4e} rel {rel:.4e}  {'OK' if ok else 'FAIL'}")
        assert ok, "q8 layout wrong"


def test_C():
    print("== Test C: reorder-undo (each recovers the logical matrix) ==")
    p = R.STATE_SIZE; g = R.N_HEADS
    # (q g p) rows: z-gate / qkv-vhalf / conv-vhalf ; out=2*g*p=4096
    W = (rng.standard_normal((2 * g * p, 512)) * 0.05).astype(np.float32)
    stored = rearrange(W, '(q g p) c -> (g q p) c', q=2, g=g, p=p)     # what 1.0.3 stores
    packed, recon = pack_q4k(stored)
    got = R._undo_qgp_rows(R.dequant_q4k_file(packed, 2 * g * p, 512), p)
    # compare undo(dequant(stored)) vs the logical order recovered from stored_recon:
    stored_recon = recon.reshape(2 * g * p, 512)
    logical_recon = rearrange(stored_recon, '(g q p) c -> (q g p) c', q=2, g=g, p=p)
    mx, rel = relerr(got, logical_recon)
    print(f"  (q g p) rows: rel {rel:.4e}  {'OK' if rel < 5e-3 else 'FAIL'}")
    assert rel < 5e-3

    # (g q p) cols: ssm_out_proj ; in=2*g*p=4096
    W = (rng.standard_normal((512, 2 * g * p)) * 0.05).astype(np.float32)
    stored = rearrange(W, 'r (q g p) -> r (g q p)', q=2, g=g, p=p)
    packed, recon = pack_q4k(stored)
    stored_recon = recon.reshape(512, 2 * g * p)
    got = R._undo_qgp_cols(R.dequant_q4k_file(packed, 512, 2 * g * p), p)
    logical_recon = rearrange(stored_recon, 'r (g q p) -> r (q g p)', q=2, g=g, p=p)
    mx, rel = relerr(got, logical_recon)
    print(f"  (g q p) cols: rel {rel:.4e}  {'OK' if rel < 5e-3 else 'FAIL'}")
    assert rel < 5e-3

    # (g q) 1-D: ssm_a / ssm_dt  (length 32)
    v = rng.standard_normal(2 * g).astype(np.float32)
    stored = rearrange(v, '(q g) -> (g q)', q=2, g=g)
    got = R._undo_qg_vec(stored)
    mx, rel = relerr(got, v)
    print(f"  (g q) vec   : rel {rel:.4e}  {'OK' if rel < 1e-6 else 'FAIL'}")
    assert rel < 1e-6


if __name__ == "__main__":
    test_A()
    test_B()
    test_C()
    print("\nALL PASS")
