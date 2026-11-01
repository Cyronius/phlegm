//===- ew.cc -----------------------------------------------------*- C++ -*-===//
//
// dit_ew's kernels (dit_ew.py): the row-wise and elementwise ops of a diffusion
// transformer step, one row element (EW_EL bf16) per call, round-to-nearest-even.
//
//   ln_mod_row     y = LN(x) * (1 + scale) + shift                   (FLUX.2 norm1/norm2/norm_out)
//                  or, R_NORM = 1, y = RMSNorm(x) * w                  (Qwen3 input/post-attn norm)
//   res_ln_mod_row x' = x + gate * y (R_UNIT_GATE: x + y);  z = the norm of x', as above
//   qk_row         per-head RMSNorm (weights wq / wk), then RoPE: FLUX.2's 4-axis interleaved
//                  pairs, or (R_ROPE = 1) Qwen3's 1-D rotate-half, theta 1e6
//   qk_moss_row    MOSS image-first four-axis, centered, half-split RoPE
//   bin_row        swiglu: silu(a) * b     euler: a + dt * b
//   situ_row       MOSS SiTU hidden/gate activation
//   un_row         silu
//   gelu_row       GELU(approximate="tanh") (Qwen Image 2.1 text projection; own program)
//
// Arithmetic: bf16 x bf16 products into fp32 accumulators, 32 lanes. aie2p has no
// native fp32 vector multiply (it is emulated, and a first fp32 version of these
// kernels ran 4-6x slower than their DMA), so a float scalar that multiplies a vector
// (rstd, dt, cos/sin) is split into bf16 hi + lo and applied as two MACs -- exact to
// fp32 rounding. Values are rounded to bf16 where diffusers' bf16 pipeline rounds
// them: the LayerNorm / RMSNorm output before the modulation / weight, the residual
// stream, 1 + scale.
//
// LN and RMSNorm run over the first W columns (a runtime parameter); columns W..EL of
// a norm's output are written as zeros. Per-dispatch parameter vectors (the modulation
// shift/scale/gate, the qk weights and RoPE tables, Euler's dt) arrive ahead of the
// rows and are copied into `par` by load_param.
//
// Accuracy notes (utilities/aie-probes/vecmath_probe.py, exp2_probe.py): aie2p's
// vector tanh is an approximation (1.4-3.5% rms below |x| = 2). SiLU is written
// 0.5x(1 + tanh(x/2)), where the leading 1 keeps tanh's error under ~0.5% of the
// result.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

#ifndef EW_EL
#define EW_EL 3072 // bf16 per row element (FLUX.2 [klein] hidden size)
#endif
#ifndef EW_NPAR
#define EW_NPAR 3 // parameter elements a core keeps
#endif
constexpr int EL = EW_EL;
constexpr int HD = 128; // head dim
constexpr float EPS = 1e-6f;

// Runtime parameter slots (dit_ew.py RTP_*).
enum {
  R_NPAR, R_CNT_LN, R_CNT_RES, R_CNT_QK, R_CNT_BIN, R_CNT_UN, R_W, R_IDX_GATE,
  R_IDX_SHIFT, R_IDX_SCALE, R_BINOP, R_ROW0, R_ROW_STEP, R_N_TXT, R_GRID_W, R_HEADS,
  R_NORM, R_UNIT_GATE, R_ROPE, R_B_QH, R_B_KH, R_GRID_H, R_AFFINE_RMS,
  R_BIAS_PERIOD, R_TANH_GATE, R_REF_ROW
};
enum { BIN_SWIGLU = 1, BIN_EULER = 2 };

// qk parameter block (make_test.py qk_params / qwen_qk_params): element 0 = wq[128],
// wk[128] (bf16); elements 1-2 = fp32 RoPE tables, as raw bits.
//   FLUX.2: FINE[p] = cos[16], sin[16] of p*f_j (p < 64), then COARSE[a] of 64a*f_j (a < 8).
//   Qwen3:  T64[a], T8[b], T1[c] = cos[64], sin[64] of (64a | 8b | c)*f_j (a, b, c < 8):
//           position 64a + 8b + c by two angle additions.
constexpr int ROPE_FINE = 0, ROPE_COARSE = 64 * 32;
constexpr int QROPE_T64 = 0, QROPE_T8 = 8 * 128, QROPE_T1 = 16 * 128;

using VB = aie::vector<bfloat16, 32>;
using AC = aie::accum<accfloat, 32>;
using VF16 = aie::vector<float, 16>;

struct Split { bfloat16 hi, lo; };
static inline Split split(float f) {
  bfloat16 hi = (bfloat16)f;
  return {hi, (bfloat16)(f - float(hi))};
}
static inline AC acc_of(VB v) {
  AC a;
  a.from_vector(v);
  return a;
}
static inline AC acc_bcast(float f) {
  AC a;
  a.from_vector(aie::broadcast<float, 32>(f));
  return a;
}
// a + v * s, s a float split into bf16 hi + lo
static inline AC mac_s(AC a, VB v, Split s) {
  return aie::mac(aie::mac(a, v, s.hi), v, s.lo);
}

// Sum and sum of squares of x[0:W] (bf16) in fp32; LayerNorm mean and 1/std, or for
// RMSNorm (rms) mean 0 and 1/rms.
static inline void ln_stats(const bfloat16 *__restrict x, int W, float &mean, float &rstd,
                            bool rms) {
  AC s = aie::zeros<accfloat, 32>(), q = aie::zeros<accfloat, 32>();
  for (int i = 0; i < W; i += 32) {
    VB v = aie::load_v<32>(x + i);
    s = aie::add(s, v);
    q = aie::mac(q, v, v);
  }
  const float inv_w = 1.0f / float(W);
  mean = rms ? 0.0f : aie::reduce_add(s.to_vector<float>()) * inv_w;
  float var = aie::reduce_add(q.to_vector<float>()) * inv_w - mean * mean;
  rstd = aie::invsqrt((var > 0.0f ? var : 0.0f) + EPS);
}

// y[0:W] = bf16((x - mean) * rstd) * onep + shift; y[W:EL] = 0. onep = bf16(1 + scale),
// or the RMSNorm weight with shift = null (no shift).
static inline void ln_apply(const bfloat16 *__restrict x, bfloat16 *__restrict y,
                            const bfloat16 *__restrict onep,
                            const bfloat16 *__restrict shift, int W, float mean,
                            float rstd) {
  const Split r = split(rstd);
  const AC off = acc_bcast(-mean * rstd);
  for (int i = 0; i < W; i += 32) {
    VB t = mac_s(off, aie::load_v<32>(x + i), r).to_vector<bfloat16>();
    AC o = shift ? aie::mac(acc_of(aie::load_v<32>(shift + i)), t, aie::load_v<32>(onep + i))
                 : aie::mul(t, aie::load_v<32>(onep + i));
    aie::store_v(y + i, o.to_vector<bfloat16>());
  }
  const VB z = aie::zeros<bfloat16, 32>();
  for (int i = W; i < EL; i += 32)
    aie::store_v(y + i, z);
}

// silu(x) = h (1 + tanh h), h = x / 2 (exact in bf16), in fp32, rounded once.
static inline VB silu32(VB x) {
  VB h = aie::mul(x, (bfloat16)0.5f).to_vector<bfloat16>();
  AC ha = acc_of(h);
  aie::vector<float, 32> hf = ha.to_vector<float>();
  VB th;
  th.insert(0, aie::tanh<bfloat16>(hf.extract<16>(0)));
  th.insert(1, aie::tanh<bfloat16>(hf.extract<16>(1)));
  return aie::mac(ha, h, th).to_vector<bfloat16>();
}

// Qwen Image 2.1 text projection uses GELU(approximate="tanh"). Keep the
// polynomial and tanh argument in fp32; the result is rounded to bf16 once.
static inline VB gelu_tanh32(VB x) {
  VB th;
  const VF16 cubic = aie::broadcast<float, 16>(0.044715f);
  const VF16 gain = aie::broadcast<float, 16>(0.7978845608028654f);
  auto xf = acc_of(x).to_vector<float>();
  for (int half = 0; half < 2; ++half) {
    VF16 a = xf.extract<16>(half);
    VF16 square = aie::mul(a, a).to_vector<float>();
    VF16 cube = aie::mul(square, a).to_vector<float>();
    VF16 inner = aie::add(a, aie::mul(cube, cubic).to_vector<float>());
    th.insert(half, aie::tanh<bfloat16>(aie::mul(inner, gain).to_vector<float>()));
  }
  VB h = aie::mul(x, (bfloat16)0.5f).to_vector<bfloat16>();
  return aie::mac(acc_of(h), h, th).to_vector<bfloat16>();
}

static inline VB tanh_bf16(VB x) {
  aie::vector<float, 32> xf = acc_of(x).to_vector<float>();
  VB y;
  y.insert(0, aie::tanh<bfloat16>(xf.extract<16>(0)));
  y.insert(1, aie::tanh<bfloat16>(xf.extract<16>(1)));
  return y;
}

// MOSS Image's SiTU(up, gate), with its two 8192-wide halves padded to three
// 3072-wide EW elements by the MOSS weight packer. This never applies SwiGLU.
static inline VB tanh32(VB x) {
  // XDNA2's tanh LUT has about 5% relative error near |x|=0.5. Use the
  // [5/4] Pade form, clamped to 4 where its absolute error is below 0.003.
  const VB z = aie::min(aie::max(x, (bfloat16)-4.0f), (bfloat16)4.0f);
  const VB z2 = aie::mul(z, z).to_vector<bfloat16>();
  const VB z4 = aie::mul(z2, z2).to_vector<bfloat16>();
  AC num = aie::mac(acc_bcast(1.0f), z2, (bfloat16)(1.0f / 9.0f));
  num = aie::mac(num, z4, (bfloat16)(1.0f / 945.0f));
  AC den = aie::mac(acc_bcast(1.0f), z2, (bfloat16)(4.0f / 9.0f));
  den = aie::mac(den, z4, (bfloat16)(1.0f / 63.0f));
  const VB scaled = aie::mul(z, num.to_vector<bfloat16>()).to_vector<bfloat16>();
  return aie::mul(scaled, aie::inv(den.to_vector<bfloat16>())).to_vector<bfloat16>();
}

static inline VB situ32(VB hidden, VB gate) {
  const VB one = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
  const VB gh = aie::mul(gate, (bfloat16)0.5f).to_vector<bfloat16>();
  const VB gq = aie::mul(gate, (bfloat16)0.25f).to_vector<bfloat16>();
  const VB hu = aie::mul(hidden, (bfloat16)0.04f).to_vector<bfloat16>();
  const VB sigmoid = aie::mul(aie::add(one, tanh32(gh)),
                              (bfloat16)0.5f).to_vector<bfloat16>();
  const VB gated = aie::mul(tanh32(gq), sigmoid).to_vector<bfloat16>();
  const VB bounded = aie::mul(tanh32(hu), (bfloat16)25.0f).to_vector<bfloat16>();
  return aie::mul(aie::mul(gated, (bfloat16)4.0f).to_vector<bfloat16>(),
                  bounded).to_vector<bfloat16>();
}

// Interleaved-pair swap: [x0, x1, x2, x3, ...] -> [x1, x0, x3, x2, ...].
static inline VB pair_swap(VB v) {
  auto u = aie::interleave_unzip(v, v, 1);
  return aie::interleave_zip(u.second, u.first, 1).first;
}

// Per-dim RoPE factors of the current token as bf16 hi + lo; sign baked into sin so
// that out = y*cos + pair_swap(y)*sin.
alignas(64) static bfloat16 s_chi[HD], s_clo[HD], s_shi[HD], s_slo[HD];
alignas(64) static float s_c[HD], s_s[HD];

// One 32-dim RoPE axis block: dims 32*axis + 2j, +1 rotate by (c_j, s_j) (fp32 16).
static inline void rope_axis(int axis, VF16 c, VF16 s) {
  const VF16 ns = aie::sub(aie::zeros<float, 16>(), s);
  auto cc = aie::interleave_zip(c, c, 1);
  auto ss = aie::interleave_zip(ns, s, 1);
  aie::store_v(s_c + 32 * axis, cc.first);
  aie::store_v(s_c + 32 * axis + 16, cc.second);
  aie::store_v(s_s + 32 * axis, ss.first);
  aie::store_v(s_s + 32 * axis + 16, ss.second);
}

static inline VF16 mulf(VF16 a, VF16 b) { return aie::mul(a, b).template to_vector<float>(); }

// RoPE factors for global token g of the joint [text; image] sequence: text token l:
// axis 3 (dims 96-127) by l; image token k: axis 1 by k / grid_w, axis 2 by k % grid_w.
// Other axes are the identity. An edit appends one reference image of the same grid
// (klein's [text | generated | reference]): image token k >= grid_w^2 is reference token
// k - grid_w^2, whose axis 0 turns by REF_T (diffusers' t = 10 for the first reference).
// Once per token; the fp32 work here is 128 values.
constexpr int REF_T = 10;
static __attribute__((noinline)) void rope_token(const float *__restrict tab, int g, int n_txt, int grid_w) {
  const VF16 one = aie::broadcast<float, 16>(1.0f), zero = aie::zeros<float, 16>();
  for (int a = 0; a < 4; a++)
    rope_axis(a, one, zero);
  if (g < n_txt) {
    const int hi = g >> 6, lo = g & 63;
    const float *ph = tab + ROPE_COARSE + hi * 32, *pl = tab + ROPE_FINE + lo * 32;
    VF16 ca = aie::load_v<16>(ph), sa = aie::load_v<16>(ph + 16);
    VF16 cb = aie::load_v<16>(pl), sb = aie::load_v<16>(pl + 16);
    // cos(a+b) = ca cb - sa sb;  sin(a+b) = sa cb + ca sb
    rope_axis(3, aie::sub(mulf(ca, cb), mulf(sa, sb)), aie::add(mulf(sa, cb), mulf(ca, sb)));
  } else {
    int k = g - n_txt;
    if (k >= grid_w * grid_w) {
      k -= grid_w * grid_w;
      const float *pt = tab + ROPE_FINE + REF_T * 32;
      rope_axis(0, aie::load_v<16>(pt), aie::load_v<16>(pt + 16));
    }
    const int h = k / grid_w, w = k - h * grid_w;
    const float *ph = tab + ROPE_FINE + h * 32, *pw = tab + ROPE_FINE + w * 32;
    rope_axis(1, aie::load_v<16>(ph), aie::load_v<16>(ph + 16));
    rope_axis(2, aie::load_v<16>(pw), aie::load_v<16>(pw + 16));
  }
  // bf16 hi (vector), then lo = fp32 - hi rounded to bf16 (scalar, 2 x 128 per token)
  for (int i = 0; i < HD; i += 16) {
    aie::accum<accfloat, 16> c, sn;
    c.from_vector(aie::load_v<16>(s_c + i));
    sn.from_vector(aie::load_v<16>(s_s + i));
    aie::store_v(s_chi + i, c.template to_vector<bfloat16>());
    aie::store_v(s_shi + i, sn.template to_vector<bfloat16>());
  }
  for (int i = 0; i < HD; i++) {
    s_clo[i] = (bfloat16)(s_c[i] - float(s_chi[i]));
    s_slo[i] = (bfloat16)(s_s[i] - float(s_shi[i]));
  }
}

// FLUX 3's published reference uses the same four 32-wide adjacent-pair axes
// as the FLUX.2 q/k kernel, but scales generated-image h/w coordinates by 0.5.
// The fine and coarse tables retain full-frequency text positions; one extra
// 32-float entry at [2304:2336] represents a half image step. This mode covers
// Image rows at or past ref_row belong to one reference image: its h restarts
// at zero and its l coordinate is -1 (the published reference-image id), the
// conjugate of the position-1 fine entry. ref_row=64 disables references.
static __attribute__((noinline)) void flux3_rope_token(
    const float *__restrict tab, int g, int n_txt, int grid_w, int ref_row) {
  const VF16 one = aie::broadcast<float, 16>(1.0f), zero = aie::zeros<float, 16>();
  for (int a = 0; a < 4; a++)
    rope_axis(a, one, zero);
  if (g < n_txt) {
    const int hi = g >> 6, lo = g & 63;
    const float *ph = tab + ROPE_COARSE + hi * 32, *pl = tab + ROPE_FINE + lo * 32;
    VF16 ca = aie::load_v<16>(ph), sa = aie::load_v<16>(ph + 16);
    VF16 cb = aie::load_v<16>(pl), sb = aie::load_v<16>(pl + 16);
    rope_axis(3, aie::sub(mulf(ca, cb), mulf(sa, sb)), aie::add(mulf(sa, cb), mulf(ca, sb)));
  } else {
    const int k = g - n_txt;
    int h = k / grid_w;
    const int w = k - h * grid_w;
    if (h >= ref_row) {
      h -= ref_row;
      const float *pl = tab + ROPE_FINE + 32;
      rope_axis(3, aie::load_v<16>(pl), aie::neg(aie::load_v<16>(pl + 16)));
    }
    const float *half = tab + ROPE_COARSE + 8 * 32;
    VF16 ch = aie::load_v<16>(half), sh = aie::load_v<16>(half + 16);
    for (int axis = 0; axis < 2; ++axis) {
      const int coord = axis ? w : h;
      const float *base = tab + ROPE_FINE + (coord >> 1) * 32;
      VF16 ca = aie::load_v<16>(base), sa = aie::load_v<16>(base + 16);
      if (coord & 1)
        rope_axis(1 + axis, aie::sub(mulf(ca, ch), mulf(sa, sh)),
                  aie::add(mulf(sa, ch), mulf(ca, sh)));
      else
        rope_axis(1 + axis, ca, sa);
    }
  }
  for (int i = 0; i < HD; i += 16) {
    aie::accum<accfloat, 16> c, sn;
    c.from_vector(aie::load_v<16>(s_c + i));
    sn.from_vector(aie::load_v<16>(s_s + i));
    aie::store_v(s_chi + i, c.template to_vector<bfloat16>());
    aie::store_v(s_shi + i, sn.template to_vector<bfloat16>());
  }
  for (int i = 0; i < HD; i++) {
    s_clo[i] = (bfloat16)(s_c[i] - float(s_chi[i]));
    s_slo[i] = (bfloat16)(s_s[i] - float(s_shi[i]));
  }
}

// Qwen Image 2.1: complex-pair RoPE with 16 frame, 56 height and 56 width
// channels. Each table has 32 positions × 64 complex pairs × (cos,sin),
// stored as BF16 in one EL=4096 parameter element. Factor p = 32a + b;
// negative centered image coordinates negate sin after factorization.
// cond > 0: an edit's published order [text 0..cond) [condition image] [text] [target
// image], both images grid_h x grid_w and n_txt text rows in all. Each image block
// freezes the frame axis at the text position before it, and the text after a block
// resumes max(grid_h, grid_w) further on (diffusers QwenImage21Rope).
static __attribute__((noinline)) void qwen21_rope_token(
    const bfloat16 *__restrict tab, int g, int n_txt, int grid_h, int grid_w, int cond) {
  const bfloat16 *__restrict fine = tab;
  const bfloat16 *__restrict coarse = tab + EL;
  int frame = g, height = g, width = g, k = -1;
  if (cond > 0) {
    const int hw = grid_h * grid_w, gap = grid_h > grid_w ? grid_h : grid_w;
    if (g >= n_txt + hw) {
      k = g - n_txt - hw;
      frame = n_txt + gap;
    } else if (g >= cond + hw) {
      frame = height = width = g - hw + gap;
    } else if (g >= cond) {
      k = g - cond;
      frame = cond;
    }
  } else if (g >= n_txt) {
    k = g - n_txt;
    frame = n_txt;
  }
  if (k >= 0) {
    height = k / grid_w - (grid_h - grid_h / 2);
    width = k % grid_w - (grid_w - grid_w / 2);
  }
  for (int pair = 0; pair < HD / 2; ++pair) {
    const int pos = pair < 8 ? frame : pair < 36 ? height : width;
    const int mag = pos < 0 ? -pos : pos;
    const int fi = (mag & 31) * HD + 2 * pair;
    const int co = (mag >> 5) * HD + 2 * pair;
    const float cf = float(fine[fi]), sf = float(fine[fi + 1]);
    const float cc = float(coarse[co]), sc = float(coarse[co + 1]);
    const float c = cf * cc - sf * sc;
    const float s = (pos < 0 ? -1.0f : 1.0f) * (sf * cc + cf * sc);
    s_c[2 * pair] = s_c[2 * pair + 1] = c;
    s_s[2 * pair] = -s;
    s_s[2 * pair + 1] = s;
  }
  for (int i = 0; i < HD; i += 16) {
    aie::accum<accfloat, 16> c, sn;
    c.from_vector(aie::load_v<16>(s_c + i));
    sn.from_vector(aie::load_v<16>(s_s + i));
    aie::store_v(s_chi + i, c.template to_vector<bfloat16>());
    aie::store_v(s_shi + i, sn.template to_vector<bfloat16>());
  }
  for (int i = 0; i < HD; ++i) {
    s_clo[i] = (bfloat16)(s_c[i] - float(s_chi[i]));
    s_slo[i] = (bfloat16)(s_s[i] - float(s_shi[i]));
  }
}

// (The qk helpers are noinline: qk_row calls each up to three times, and inlined copies
// overflow the core's 16 KB program memory.)
// Qwen3 rotate-half RoPE factors for position g: cos/sin of g*f_j, j < 64, into
// s_chi/s_clo (cos) and s_shi/s_slo (sin), dims 0..63 (the pair (j, j+64) shares them).
static __attribute__((noinline)) void qwen_rope_token(const float *__restrict tab, int g) {
  const float *pa = tab + QROPE_T64 + (g >> 6) * 128;
  const float *pb = tab + QROPE_T8 + ((g >> 3) & 7) * 128;
  const float *pc = tab + QROPE_T1 + (g & 7) * 128;
  for (int i = 0; i < 64; i += 16) {
    VF16 ca = aie::load_v<16>(pa + i), sa = aie::load_v<16>(pa + 64 + i);
    VF16 cb = aie::load_v<16>(pb + i), sb = aie::load_v<16>(pb + 64 + i);
    VF16 c1 = aie::sub(mulf(ca, cb), mulf(sa, sb)), s1 = aie::add(mulf(sa, cb), mulf(ca, sb));
    VF16 cc = aie::load_v<16>(pc + i), sc = aie::load_v<16>(pc + 64 + i);
    aie::store_v(s_c + i, aie::sub(mulf(c1, cc), mulf(s1, sc)));
    aie::store_v(s_s + i, aie::add(mulf(s1, cc), mulf(c1, sc)));
  }
  for (int i = 0; i < 64; i += 16) {
    aie::accum<accfloat, 16> c, sn;
    c.from_vector(aie::load_v<16>(s_c + i));
    sn.from_vector(aie::load_v<16>(s_s + i));
    aie::store_v(s_chi + i, c.template to_vector<bfloat16>());
    aie::store_v(s_shi + i, sn.template to_vector<bfloat16>());
  }
  for (int i = 0; i < 64; i++) {
    s_clo[i] = (bfloat16)(s_c[i] - float(s_chi[i]));
    s_slo[i] = (bfloat16)(s_s[i] - float(s_shi[i]));
  }
}

// MOSS uses four 16-wide axes in the first half of each 128-wide head, and
// rotates against the corresponding dimensions in the second half. The r axis
// is zero for text-to-image. A compact bf16 table stores 65 coarse positions,
// 64 fine positions and the two centered image-coordinate offsets. It fits in
// the existing two 3072-wide parameter elements after the q/k norm weights.
constexpr int M_COARSE = 0, M_FINE = 65 * 32, M_HOFF = M_FINE + 64 * 32,
              M_WOFF = M_HOFF + 32;
static inline VF16 moss_float16(const bfloat16 *__restrict p) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(p));
  return a.to_vector<float>();
}
static inline void moss_angle(VF16 ca, VF16 sa, VF16 cb, VF16 sb, VF16 &c, VF16 &s) {
  c = aie::sub(mulf(ca, cb), mulf(sa, sb));
  s = aie::add(mulf(sa, cb), mulf(ca, sb));
}
static __attribute__((noinline)) void moss_rope_token(
    const bfloat16 *__restrict tab, int g, int n_txt, int grid_h, int grid_w) {
  const VF16 one = aie::broadcast<float, 16>(1.0f), zero = aie::zeros<float, 16>();
  for (int i = 0; i < 64; i += 16) {
    aie::store_v(s_c + i, one);
    aie::store_v(s_s + i, zero);
  }
  const int image_tokens = grid_h * grid_w;
  const bool image = g < image_tokens;
  const int f = image ? n_txt : g - image_tokens;
  const bfloat16 *coarse = tab + M_COARSE + (f >> 6) * 32;
  const bfloat16 *fine = tab + M_FINE + (f & 63) * 32;
  VF16 c, s;
  moss_angle(moss_float16(coarse), moss_float16(coarse + 16),
             moss_float16(fine), moss_float16(fine + 16), c, s);
  aie::store_v(s_c + 16, c);
  aie::store_v(s_s + 16, s);
  if (image) {
    const int h = g / grid_w, w = g - h * grid_w;
    for (int axis = 0; axis < 2; ++axis) {
      const bfloat16 *coord = tab + M_FINE + (axis ? w : h) * 32;
      const bfloat16 *shift = tab + (axis ? M_WOFF : M_HOFF);
      moss_angle(moss_float16(coord), moss_float16(coord + 16),
                 moss_float16(shift), moss_float16(shift + 16), c, s);
      aie::store_v(s_c + 32 + axis * 16, c);
      aie::store_v(s_s + 32 + axis * 16, s);
    }
  }
  for (int i = 0; i < 64; i += 16) {
    aie::accum<accfloat, 16> ca, sa;
    ca.from_vector(aie::load_v<16>(s_c + i));
    sa.from_vector(aie::load_v<16>(s_s + i));
    aie::store_v(s_chi + i, ca.to_vector<bfloat16>());
    aie::store_v(s_shi + i, sa.to_vector<bfloat16>());
  }
  for (int i = 0; i < 64; ++i) {
    s_clo[i] = (bfloat16)(s_c[i] - float(s_chi[i]));
    s_slo[i] = (bfloat16)(s_s[i] - float(s_shi[i]));
  }
}

// RMSNorm (weight w) then Qwen3 rotate-half RoPE of one row of `heads` heads:
// out[j] = t[j] c_j - t[j+64] s_j,  out[j+64] = t[j+64] c_j + t[j] s_j.
static __attribute__((noinline)) void qk_heads_half(const bfloat16 *__restrict x, bfloat16 *__restrict y,
                                 const bfloat16 *__restrict w, int heads) {
  for (int h = 0; h < heads; h++) {
    const bfloat16 *__restrict xh = x + h * HD;
    bfloat16 *__restrict yh = y + h * HD;
    AC q = aie::zeros<accfloat, 32>();
    for (int i = 0; i < HD; i += 32) {
      VB v = aie::load_v<32>(xh + i);
      q = aie::mac(q, v, v);
    }
    const Split rs = split(aie::invsqrt(aie::reduce_add(q.to_vector<float>()) *
                                            (1.0f / HD) + EPS));
    for (int i = 0; i < 64; i += 32) {
      VB lo = aie::mul(mac_s(aie::zeros<accfloat, 32>(), aie::load_v<32>(xh + i), rs)
                           .to_vector<bfloat16>(), aie::load_v<32>(w + i)).to_vector<bfloat16>();
      VB hi = aie::mul(mac_s(aie::zeros<accfloat, 32>(), aie::load_v<32>(xh + 64 + i), rs)
                           .to_vector<bfloat16>(), aie::load_v<32>(w + 64 + i)).to_vector<bfloat16>();
      VB ch = aie::load_v<32>(s_chi + i), cl = aie::load_v<32>(s_clo + i);
      VB sh = aie::load_v<32>(s_shi + i), sl = aie::load_v<32>(s_slo + i);
      AC o_lo = aie::mac(aie::mul(lo, ch), lo, cl);
      o_lo = aie::msc(aie::msc(o_lo, hi, sh), hi, sl);
      AC o_hi = aie::mac(aie::mul(hi, ch), hi, cl);
      o_hi = aie::mac(aie::mac(o_hi, lo, sh), lo, sl);
      aie::store_v(yh + i, o_lo.to_vector<bfloat16>());
      aie::store_v(yh + 64 + i, o_hi.to_vector<bfloat16>());
    }
  }
}

static inline void copy_heads(const bfloat16 *__restrict x, bfloat16 *__restrict y, int heads) {
  for (int i = 0; i < heads * HD; i += 32)
    aie::store_v(y + i, aie::load_v<32>(x + i));
}

// RMSNorm (weight w) then RoPE of one row of `heads` heads.
static __attribute__((noinline)) void qk_heads(const bfloat16 *__restrict x, bfloat16 *__restrict y,
                            const bfloat16 *__restrict w, int heads) {
  for (int h = 0; h < heads; h++) {
    const bfloat16 *__restrict xh = x + h * HD;
    bfloat16 *__restrict yh = y + h * HD;
    AC q = aie::zeros<accfloat, 32>();
    for (int i = 0; i < HD; i += 32) {
      VB v = aie::load_v<32>(xh + i);
      q = aie::mac(q, v, v);
    }
    const Split rs = split(aie::invsqrt(aie::reduce_add(q.to_vector<float>()) *
                                            (1.0f / HD) + EPS));
    for (int i = 0; i < HD; i += 32) {
      VB u = mac_s(aie::zeros<accfloat, 32>(), aie::load_v<32>(xh + i), rs)
                 .to_vector<bfloat16>();
      VB t = aie::mul(u, aie::load_v<32>(w + i)).to_vector<bfloat16>();
      VB ts = pair_swap(t);
      AC o = aie::mul(t, aie::load_v<32>(s_chi + i));
      o = aie::mac(o, t, aie::load_v<32>(s_clo + i));
      o = aie::mac(o, ts, aie::load_v<32>(s_shi + i));
      o = aie::mac(o, ts, aie::load_v<32>(s_slo + i));
      aie::store_v(yh + i, o.to_vector<bfloat16>());
    }
  }
}

static inline void zero_tail(bfloat16 *__restrict y, int from) {
  const VB z = aie::zeros<bfloat16, 32>();
  for (int i = from; i < EL; i += 32)
    aie::store_v(y + i, z);
}

extern "C" {

#ifdef EW_NOP // timing builds only (DE_NOP=1): every row kernel returns at once
#define SET_ROUNDING() return
#else
#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)
#endif

// par[slot] = e, slot = i + shift, dropped outside [0, EW_NPAR): the i-th parameter
// element a core receives (dit_ew.py streams p[-1], p[0], p[0], .., p[n] to a pair).
// For the LN ops the scale vector is stored as bf16(1 + scale), as diffusers rounds it.
void load_param(bfloat16 *__restrict e, bfloat16 *__restrict par, int32_t i, int32_t shift,
                int32_t *__restrict rtp) {
  SET_ROUNDING();
  const int slot = i + shift;
  if (slot < 0 || slot >= EW_NPAR)
    return;
  bfloat16 *__restrict d = par + slot * EL;
  const bool onep = (rtp[R_CNT_LN] | rtp[R_CNT_RES]) && !rtp[R_NORM] &&
                    slot == rtp[R_IDX_SCALE];
  const VB one = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
  for (int k = 0; k < EL; k += 32) {
    VB v = aie::load_v<32>(e + k);
    aie::store_v(d + k, onep ? aie::add(v, one) : v);
  }
}

void ln_mod_row(bfloat16 *__restrict x, bfloat16 *__restrict y, bfloat16 *__restrict par,
                int32_t *__restrict rtp) {
  SET_ROUNDING();
  const int W = rtp[R_W];
  const bool rms = rtp[R_NORM];
  float mean, rstd;
  ln_stats(x, W, mean, rstd, rms);
  ln_apply(x, y, par + rtp[R_IDX_SCALE] * EL,
           rms && !rtp[R_AFFINE_RMS] ? nullptr : par + rtp[R_IDX_SHIFT] * EL, W,
           mean, rstd);
}

// xo = bf16(x + gate * yb) (the residual stream); z = LN(xo)(1 + scale) + shift.
void res_ln_mod_row(bfloat16 *__restrict x, bfloat16 *__restrict yb, bfloat16 *__restrict z,
                    bfloat16 *__restrict xo, bfloat16 *__restrict par,
                    int32_t *__restrict rtp) {
  SET_ROUNDING();
  const int W = rtp[R_W];
  const bool rms = rtp[R_NORM];
  const bfloat16 *__restrict g = par + rtp[R_IDX_GATE] * EL;
  const VB one = aie::broadcast<bfloat16, 32>((bfloat16)1.0f);
  const bool unit = rtp[R_UNIT_GATE];
  AC s = aie::zeros<accfloat, 32>(), q = aie::zeros<accfloat, 32>();
  for (int i = 0; i < W; i += 32) {
    VB gate = unit ? one : aie::load_v<32>(g + i);
    if (rtp[R_TANH_GATE]) gate = tanh_bf16(gate);
    VB r = aie::mac(acc_of(aie::load_v<32>(x + i)), gate,
                    aie::load_v<32>(yb + i)).to_vector<bfloat16>();
    aie::store_v(xo + i, r);
    s = aie::add(s, r);
    q = aie::mac(q, r, r);
  }
  const VB zz = aie::zeros<bfloat16, 32>();
  for (int i = W; i < EL; i += 32)
    aie::store_v(xo + i, zz);
  const float inv_w = 1.0f / float(W);
  const float mean = rms ? 0.0f : aie::reduce_add(s.to_vector<float>()) * inv_w;
  const float var = aie::reduce_add(q.to_vector<float>()) * inv_w - mean * mean;
  const float rstd = aie::invsqrt((var > 0.0f ? var : 0.0f) + EPS);
  ln_apply(xo, z, par + rtp[R_IDX_SCALE] * EL,
           rms && !rtp[R_AFFINE_RMS] ? nullptr : par + rtp[R_IDX_SHIFT] * EL, W,
           mean, rstd);
}

// Two row elements of token rtp[ROW0] + t * rtp[ROW_STEP], RMSNorm per head then RoPE.
// Element A: rtp[HEADS] heads, all q (weight wq). Element B: rtp[B_QH] q heads, then
// rtp[B_KH] k heads (weight wk), the rest of its 24 copied through (Qwen3's v heads
// share the fused q|k|v row). FLUX.2: A = q, B = k (B_QH 0, B_KH 24).
void qk_row(bfloat16 *__restrict a, bfloat16 *__restrict b, bfloat16 *__restrict ao,
            bfloat16 *__restrict bo, bfloat16 *__restrict par, int32_t *__restrict rtp,
            int32_t t) {
  SET_ROUNDING();
  const int g = rtp[R_ROW0] + t * rtp[R_ROW_STEP];
  const float *tab = (const float *)(par + EL);
  const int ha = rtp[R_HEADS], bq = rtp[R_B_QH], bk = rtp[R_B_KH];
  if (rtp[R_ROPE]) {
    qwen_rope_token(tab, g);
    qk_heads_half(a, ao, par, ha);
    qk_heads_half(b, bo, par, bq);
    qk_heads_half(b + bq * HD, bo + bq * HD, par + HD, bk);
  } else {
    rope_token(tab, g, rtp[R_N_TXT], rtp[R_GRID_W]);
    qk_heads(a, ao, par, ha);
    qk_heads(b, bo, par, bq);
    qk_heads(b + bq * HD, bo + bq * HD, par + HD, bk);
  }
  zero_tail(ao, ha * HD);
  copy_heads(b + (bq + bk) * HD, bo + (bq + bk) * HD, EL / HD - bq - bk);
}

void qk_moss_row(bfloat16 *__restrict q, bfloat16 *__restrict k,
                 bfloat16 *__restrict qo, bfloat16 *__restrict ko,
                 bfloat16 *__restrict par, int32_t *__restrict rtp, int32_t t) {
  SET_ROUNDING();
  const int g = rtp[R_ROW0] + t * rtp[R_ROW_STEP];
  if (g >= rtp[R_GRID_H] * rtp[R_GRID_W] + rtp[R_N_TXT]) {
    const VB zero = aie::zeros<bfloat16, 32>();
    for (int i = 0; i < EL; i += 32) {
      aie::store_v(qo + i, zero);
      aie::store_v(ko + i, zero);
    }
    return;
  }
  moss_rope_token(par + EL, g, rtp[R_N_TXT], rtp[R_GRID_H], rtp[R_GRID_W]);
  qk_heads_half(q, qo, par, rtp[R_HEADS]);
  qk_heads_half(k, ko, par + HD, rtp[R_HEADS]);
}

// Kept separate from qk_row so the FLUX 3 specialization fits in AIE program
// memory without linking the FLUX.2 and Qwen RoPE paths into each core.
void qk_flux3_row(bfloat16 *__restrict q, bfloat16 *__restrict k,
                  bfloat16 *__restrict qo, bfloat16 *__restrict ko,
                  bfloat16 *__restrict par, int32_t *__restrict rtp, int32_t t) {
  SET_ROUNDING();
  const int g = rtp[R_ROW0] + t * rtp[R_ROW_STEP];
  if (g >= rtp[R_N_TXT] + rtp[R_GRID_H] * rtp[R_GRID_W]) {
    const VB zero = aie::zeros<bfloat16, 32>();
    for (int i = 0; i < EL; i += 32) {
      aie::store_v(qo + i, zero);
      aie::store_v(ko + i, zero);
    }
    return;
  }
  flux3_rope_token((const float *)(par + EL), g, rtp[R_N_TXT], rtp[R_GRID_W], rtp[R_REF_ROW]);
  qk_heads(q, qo, par, rtp[R_HEADS]);
  qk_heads(k, ko, par + HD, rtp[R_HEADS]);
}

// Kept separate from the other q/k paths to stay within core program memory.
void qk_qwen21_row(bfloat16 *__restrict q, bfloat16 *__restrict k,
                   bfloat16 *__restrict qo, bfloat16 *__restrict ko,
                   bfloat16 *__restrict par, int32_t *__restrict rtp, int32_t t) {
  SET_ROUNDING();
  const int g = rtp[R_ROW0] + t * rtp[R_ROW_STEP];
  const int cond = rtp[R_REF_ROW];                 // first condition-image row; 0: none
  if (g >= rtp[R_N_TXT] + rtp[R_GRID_H] * rtp[R_GRID_W] * (cond > 0 ? 2 : 1)) {
    const VB zero = aie::zeros<bfloat16, 32>();
    for (int i = 0; i < EL; i += 32) {
      aie::store_v(qo + i, zero);
      aie::store_v(ko + i, zero);
    }
    return;
  }
  qwen21_rope_token(par + EL, g, rtp[R_N_TXT], rtp[R_GRID_H], rtp[R_GRID_W], cond);
  qk_heads(q, qo, par, rtp[R_HEADS]);
  qk_heads(k, ko, par + HD, rtp[R_HEADS]);
}

void bin_row(bfloat16 *__restrict a, bfloat16 *__restrict b, bfloat16 *__restrict y,
             bfloat16 *__restrict par, int32_t *__restrict rtp) {
  SET_ROUNDING();
  if (rtp[R_BINOP] == BIN_SWIGLU) {
    for (int i = 0; i < EL; i += 32)
      aie::store_v(y + i, aie::mul(silu32(aie::load_v<32>(a + i)), aie::load_v<32>(b + i))
                              .to_vector<bfloat16>());
  } else { // BIN_EULER: dt is par[0:2] as an fp32
    const Split dt = split(*(const float *)par);
    for (int i = 0; i < EL; i += 32)
      aie::store_v(y + i, mac_s(acc_of(aie::load_v<32>(a + i)), aie::load_v<32>(b + i), dt)
                              .to_vector<bfloat16>());
  }
}

// Keep MOSS's activation in a separate entry point so a SiTU specialization
// does not link the unrelated q/k, norm, SwiGLU and Euler row programs.
void situ_row(bfloat16 *__restrict a, bfloat16 *__restrict b, bfloat16 *__restrict y) {
  SET_ROUNDING();
  for (int i = 0; i < EL; i += 32)
    aie::store_v(y + i, situ32(aie::load_v<32>(a + i), aie::load_v<32>(b + i)));
}

void un_row(bfloat16 *__restrict a, bfloat16 *__restrict y, int32_t *__restrict rtp) {
  SET_ROUNDING();
  for (int i = 0; i < EL; i += 32)
    aie::store_v(y + i, silu32(aie::load_v<32>(a + i)));
}

// GELU-tanh is its own entry point, linked only by "gelu_tanh" streams. Inlined
// into un_row it added ~0.9 KB to every generic stream's core program, which then
// overflowed the 16 KB program memory (15,856 -> 16,880 bytes for a FLUX 3 norm).
void gelu_row(bfloat16 *__restrict a, bfloat16 *__restrict y) {
  SET_ROUNDING();
  for (int i = 0; i < EL; i += 32)
    aie::store_v(y + i, gelu_tanh32(aie::load_v<32>(a + i)));
}

// Add one of up to three per-element bias vectors to each row element. The
// stream may have a larger physical row stride (for a half of an FFN output).
void bias_row(bfloat16 *__restrict a, bfloat16 *__restrict y,
              bfloat16 *__restrict par, int32_t *__restrict rtp, int32_t t) {
  SET_ROUNDING();
  const int period = rtp[R_BIAS_PERIOD];
  const int element = (rtp[R_ROW0] + t * rtp[R_ROW_STEP]) % period;
  const bfloat16 *__restrict bias = par + element * EL;
  for (int i = 0; i < EL; i += 32)
    aie::store_v(y + i, aie::add(aie::load_v<32>(a + i),
                                 aie::load_v<32>(bias + i)));
}

} // extern "C"
