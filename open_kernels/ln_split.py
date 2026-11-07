r"""The split norm helper: the same stages, for a width the fused one cannot hold.

The fused helper (designs/ln/ln_y.cc, ln_xn.cc) takes five input elements at once --
the residual's two halves, the norm weights and the added term's two halves -- and
holds them in the norm core's L1 while it works. At HID 5120 those five 10 KB
elements plus the output element and the stack do not fit the core's 64 KB, so the
helper STREAMS instead: the residual add goes half by half (ln_add2), the sum goes
to DDR, and the norm reads it back (ln_nr). Same arithmetic, same bits -- ln_add2
forms the sum exactly as ln_y does.

Two designs need it: the MoE / Qwen3.5 tail (designs/layer_x/xcommon.py) and the
dense one (designs/dense/dx.py). The bodies live here so the two cannot drift: a
helper that adds the residual one way in one design and another way in the other
is a bug waiting for the width that reaches both.
"""
from __future__ import annotations


def split_stages(ain, aout, f_nr, f_add2, f_nr32=None, entry_fp32=False):
    """Stages 1 and 2 of the split norm helper (recipes/qwen36moe.py norm_split):

      1. layer-entry norm                 [x0 x1 lnw] -> [xn]
      2. res_i = x_i + a_i, i = 0, 1      [x_i a_i]   -> [res_i]     (res goes to DDR)
         xm = post_attention_norm(res)    [res0 res1 w] -> [xm]      (res read back)
    Three input elements at most, so the fifo is 3 deep and 4 elements are ever held."""
    e = ain.acquire(3)
    if entry_fp32:
        for i in range(2):
            o = aout.acquire(1)
            f_nr32(e[0], e[1], e[2], o, i)
            aout.release(1)
    else:
        o = aout.acquire(1)
        f_nr(e[0], e[1], e[2], o)
        aout.release(1)
    ain.release(3)
    for _ in range(2):
        e = ain.acquire(2)
        o = aout.acquire(1)
        f_add2(e[0], e[1], o)
        aout.release(1)
        ain.release(2)
    e = ain.acquire(3)
    if f_nr32 is not None:
        for i in range(2):
            o = aout.acquire(1)
            f_nr32(e[0], e[1], e[2], o, i)
            aout.release(1)
    else:
        o = aout.acquire(1)
        f_nr(e[0], e[1], e[2], o)
        aout.release(1)
    ain.release(3)


def ln_split_body(ain, aout, f_nr, f_add2):
    """... 3. xres_i = res_i + out2_i         [res_i out2_i] -> [xres_i]"""
    split_stages(ain, aout, f_nr, f_add2)
    for _ in range(2):
        e = ain.acquire(2)
        o = aout.acquire(1)
        f_add2(e[0], e[1], o)
        aout.release(1)
        ain.release(2)


def ln_split_body3(ain, aout, f_nr, f_add2, f_add3, f_nr32=None, entry_fp32=False):
    """... 3. xres_i = res_i + (out2_i + out2b_i), the two down pieces' outputs
    (recipes/qwen36moe.py down_split)   [res_i out2_i out2b_i] -> [xres_i]"""
    split_stages(ain, aout, f_nr, f_add2, f_nr32, entry_fp32)
    for _ in range(2):
        e = ain.acquire(3)
        o = aout.acquire(1)
        f_add3(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)


# ---- the quarter form (recipes/dense.py norm_quarter): HID-byte elements, for a width whose split
# helper still does not fit the norm core (Nanbeige4.3's 7168). x / a / res arrive as four fp32
# quarters, the norm weights and xn as two bf16 halves. ln_add2 / ln_add3 are the split helper's
# own TUs compiled at LN_N = HID / 2, so their "half" is a quarter of HID; ln_nrq writes one xn
# half per call from all four quarters (designs/ln/ln_nrq.cc). Same arithmetic, same bits.

def quarter_norm(ain, aout, f_nrq):
    """[w0 x0 x1 x2 x3] -> xn0, then w0 out and w1 in: [x0 x1 x2 x3 w1] -> xn1. Five inputs held."""
    e = ain.acquire(5)
    o = aout.acquire(1)
    f_nrq(e[0], e[1], e[2], e[3], e[4], o, 0)
    aout.release(1)
    ain.release(1)                     # w0, the oldest
    e = ain.acquire(5)                 # the four quarters still held, then w1
    o = aout.acquire(1)
    f_nrq(e[4], e[0], e[1], e[2], e[3], o, 1)
    aout.release(1)
    ain.release(5)


def quarter_adds(ain, aout, f_add, n_in):
    """y_q = x_q + a_q (+ b_q) for the four quarters: n_in inputs per quarter, one output."""
    for _ in range(4):
        e = ain.acquire(n_in)
        o = aout.acquire(1)
        if n_in == 2:
            f_add(e[0], e[1], o)
        else:
            f_add(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(n_in)


def ln_quarter_body(ain, aout, f_nrq, f_add2):
    """1. xn = norm(x); 2. res = x + out, xm = norm(res); 3. xres = res + out2."""
    quarter_norm(ain, aout, f_nrq)
    quarter_adds(ain, aout, f_add2, 2)
    quarter_norm(ain, aout, f_nrq)
    quarter_adds(ain, aout, f_add2, 2)


def ln_quarter_body3(ain, aout, f_nrq, f_add2, f_add3):
    """... 3. xres = res + (out2 + out2b), the two down pieces' outputs."""
    quarter_norm(ain, aout, f_nrq)
    quarter_adds(ain, aout, f_add2, 2)
    quarter_norm(ain, aout, f_nrq)
    quarter_adds(ain, aout, f_add3, 3)
