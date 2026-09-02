#pragma once
//===- dn_glue.h -------------------------------------------*- C++ -*-===//
//
// Linear-attention layer glue around the DeltaNet step (decode, T = 1):
//
//   alpha  = xn @ Wa            (2048 x 32 bf16)  -> decay[h] = exp(A[h] * softplus(alpha[h] + dt_bias[h]))
//   betal  = xn @ Wb            (2048 x 32 bf16)  -> beta[h]  = sigmoid(betal[h])
//   c      = silu(w0*s0 + w1*s1 + w2*s2 + w3*qkv)  depthwise conv k=4 over
//            [state rows 0..2, this token's qkv], 8192 channels
//   q, k   = L2-normalise per 128-head (channels 0..2047 / 2048..4095)
//   v      = c[4096..8191]
//   state' = [s1, s2, bf16(qkv)]
//   record[h] (fp32[512]) = [k[h/2] | q[h/2] | v[h] | decay[h] @384 | beta[h] @385]
//
// Reference: tools/kernel-interp/decode_step.py linear_decode. Idioms:
// sigmoid(x) = (tanh(x/2)+1)/2 with aie::tanh<bfloat16> (LLMNpuTest); every
// fp32 x anything product is a bf16 hi/lo split (no fp32 vector multiply).
//
// Channel tiles of 1024 (8 tiles): tiles 0-1 = q heads, 2-3 = k heads, 4-7 = v.

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

static constexpr unsigned kHid = 2048;
static constexpr unsigned kNHead = 32;
static constexpr unsigned kHD = 128;
static constexpr unsigned kTile = 1024;      // conv channels per tile
static constexpr unsigned kV = 32;

using v32f = aie::vector<float, kV>;
using v32b = aie::vector<bfloat16, kV>;
using accf32 = aie::accum<accfloat, kV>;

static inline void split32(const v32f &v, v32b &h, v32b &l) {
  accf32 a;
  a.from_vector(v);
  h = a.template to_vector<bfloat16>();
  l = aie::sub(a, h).template to_vector<bfloat16>();
}

// acc += a(fp32 vec) * s(bf16 hi/lo scalar)
static inline accf32 mac_vs(accf32 acc, const v32f &a, bfloat16 sh, bfloat16 sl) {
  v32b ah, al;
  split32(a, ah, al);
  acc = aie::mac(acc, ah, sh);
  acc = aie::mac(acc, ah, sl);
  acc = aie::mac(acc, al, sh);
  return acc;
}

// acc += a(fp32 vec) * b(bf16 vec)
static inline accf32 mac_vv(accf32 acc, const v32f &a, const v32b &b) {
  v32b ah, al;
  split32(a, ah, al);
  acc = aie::mac(acc, ah, b);
  acc = aie::mac(acc, al, b);
  return acc;
}

// ---- fp32 vector arithmetic on top of bf16 MACs (3 cross terms, ~2^-16 rel)
static inline v32f fmul32(const v32f &a, const v32f &b) {
  v32b ah, al, bh, bl;
  split32(a, ah, al);
  split32(b, bh, bl);
  accf32 acc = aie::mul(ah, bh);
  acc = aie::mac(acc, ah, bl);
  acc = aie::mac(acc, al, bh);
  return acc.template to_vector<float>();
}
static inline v32f fadd32(const v32f &a, const v32f &b) {
  accf32 x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::add(x, y).template to_vector<float>();
}
static inline v32f fsub32(const v32f &a, const v32f &b) {
  accf32 x, y;
  x.from_vector(a);
  y.from_vector(b);
  return aie::sub(x, y).template to_vector<float>();
}

// exp(x) to ~1e-7 relative: 2^(x*log2e) = 2^n * 2^f, |f| <= 0.5, degree-6 poly.
// The hardware tanh/exp2 are LUT linear approximations (~1e-2), unusable for a
// sigmoid that feeds an L2-normalised q/k.
static inline v32f vexp32(v32f x) {
  x = aie::max(x, aie::broadcast<float, kV>(-87.0f));
  x = aie::min(x, aie::broadcast<float, kV>(88.0f));
  const v32f t = fmul32(x, aie::broadcast<float, kV>(1.44269504f));
  const aie::vector<int32_t, kV> n = aie::to_fixed<int32_t>(t, 0);      // round (conv_even)
  const v32f nf = aie::to_float<float>(n, 0);
  const v32f f = fsub32(t, nf);                                         // [-0.5, 0.5]
  v32f p = aie::broadcast<float, kV>(1.54035304e-4f);
  p = fadd32(fmul32(p, f), aie::broadcast<float, kV>(1.33335581e-3f));
  p = fadd32(fmul32(p, f), aie::broadcast<float, kV>(9.61812911e-3f));
  p = fadd32(fmul32(p, f), aie::broadcast<float, kV>(5.55041087e-2f));
  p = fadd32(fmul32(p, f), aie::broadcast<float, kV>(2.40226507e-1f));
  p = fadd32(fmul32(p, f), aie::broadcast<float, kV>(6.93147181e-1f));
  p = fadd32(fmul32(p, f), aie::broadcast<float, kV>(1.0f));
  const aie::vector<int32_t, kV> bits =
      aie::upshift(aie::add(n, aie::broadcast<int32_t, kV>(127)), 23);
  const v32f scale = bits.template cast_to<float>();                    // exact power of two
  accf32 s;
  s.from_vector(scale);
  const v32b sb = s.template to_vector<bfloat16>();                     // exact
  v32b ph, pl;
  split32(p, ph, pl);
  accf32 r = aie::mul(ph, sb);
  r = aie::mac(r, pl, sb);
  return r.template to_vector<float>();
}

// 1/d to fp32: hardware inv seed + two Newton steps r = r (2 - d r).
static inline v32f vrecip32(const v32f &d) {
  v32f r = aie::inv(d);
  const v32f two = aie::broadcast<float, kV>(2.0f);
  r = fmul32(r, fsub32(two, fmul32(d, r)));
  r = fmul32(r, fsub32(two, fmul32(d, r)));
  return r;
}

static inline v32f vsigmoid32(const v32f &x) {
  const v32f e = vexp32(aie::neg(x));
  return vrecip32(fadd32(e, aie::broadcast<float, kV>(1.0f)));
}

// ---- alpha/beta projection tile: W element = kAbRows rows x 32 bf16 (4 KB); acc[32] += x[rows] @ W
static constexpr unsigned kAbRows = 64;
static inline void glue_ab_tile(const bfloat16 *__restrict W, const bfloat16 *__restrict xn,
                                float *__restrict acc, unsigned tile, bool first) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 a;
  if (first)
    a = aie::zeros<accfloat, kV>();
  else
    a.from_vector(aie::load_v<kV>(acc));
  const bfloat16 *x = xn + tile * kAbRows;
#pragma clang loop unroll(disable)
  for (unsigned r = 0; r < kAbRows; ++r)
    a = aie::mac(a, aie::load_v<kV>(W + r * kV), x[r]);
  aie::store_v(acc, a.template to_vector<float>());
}

// ---- scalar transcendental helpers (32 values per layer: scalar float is slow
// but this is 64 evaluations). Own implementations: no libm dependency.
static inline float sexp(float x) {
  if (x > 88.f) x = 88.f;
  if (x < -87.f) x = -87.f;
  // 2^(x*log2e): n = round, f in [-0.5, 0.5]
  const float t = x * 1.44269504f;
  const int n = (int)(t + (t >= 0.f ? 0.5f : -0.5f));
  const float f = (t - (float)n) * 0.69314718f;      // ln2 * frac -> exp(f), |f| <= 0.347
  // exp(f) by degree-6 Taylor (error < 1e-7 on this range)
  float p = 1.f + f * (1.f + f * (0.5f + f * (0.166666667f + f * (0.0416666667f + f * (0.00833333333f + f * 0.00138888889f)))));
  union { float f; uint32_t u; } s;
  s.u = (uint32_t)(n + 127) << 23;
  return p * s.f;
}
static inline float slog(float x) {                 // natural log, x > 0
  union { float f; uint32_t u; } s;
  s.f = x;
  int e = (int)((s.u >> 23) & 0xFF) - 127;
  s.u = (s.u & 0x007FFFFFu) | 0x3F800000u;           // mantissa in [1, 2)
  float m = s.f;
  if (m > 1.41421356f) { m *= 0.5f; e += 1; }        // m in [0.707, 1.414]
  const float y = (m - 1.f) / (m + 1.f);             // atanh series: ln m = 2(y + y^3/3 + y^5/5 + ...)
  const float y2 = y * y;
  const float l = 2.f * y * (1.f + y2 * (0.333333333f + y2 * (0.2f + y2 * (0.142857143f + y2 * 0.111111111f))));
  return l + (float)e * 0.69314718f;
}
static inline float ssoftplus(float u) { return u > 20.f ? u : slog(1.f + sexp(u)); }
static inline float ssigmoid(float u) { return 1.f / (1.f + sexp(-u)); }

// ---- decay/beta from the two projections
static inline void glue_small(const float *__restrict small /* A[32] @0, dt_bias[32] @32 */,
                              const float *__restrict acc_a, const float *__restrict acc_b,
                              float *__restrict decay, float *__restrict beta) {
  for (unsigned h = 0; h < kNHead; ++h) {
    decay[h] = sexp(small[h] * ssoftplus(acc_a[h] + small[32 + h]));
    beta[h] = ssigmoid(acc_b[h]);
  }
}

// ---- conv tile t: c = silu(conv), new state rows, q/k L2 norm
static inline void glue_conv_tile(const float *__restrict q0, const float *__restrict q1,
                                  const bfloat16 *__restrict s0, const bfloat16 *__restrict s1,
                                  const bfloat16 *__restrict s2, const bfloat16 *__restrict w01,
                                  const bfloat16 *__restrict w23,
                                  bfloat16 *__restrict ns0, bfloat16 *__restrict ns1,
                                  bfloat16 *__restrict ns2, float *__restrict qk,
                                  float *__restrict vt, unsigned t) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *__restrict w = w01;                // rows 0,1 (4 KB element)
  const bfloat16 *__restrict w2 = w23;               // rows 2,3
  (void)w2;
  const v32b half = aie::broadcast<bfloat16, kV>((bfloat16)0.5f);
  const v32b one = aie::broadcast<bfloat16, kV>((bfloat16)1.0f);
  float *__restrict dst = (t < 4) ? (qk + t * kTile) : vt;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kTile; j += kV) {
    const float *qp = (j < 512) ? (q0 + j) : (q1 + (j - 512));
    const v32f x = aie::load_v<kV>(qp);
    const v32b s0v = aie::load_v<kV>(s0 + j);
    const v32b s1v = aie::load_v<kV>(s1 + j);
    const v32b s2v = aie::load_v<kV>(s2 + j);
    accf32 a = aie::mul(aie::load_v<kV>(w + j), s0v);
    a = aie::mac(a, aie::load_v<kV>(w + kTile + j), s1v);
    a = aie::mac(a, aie::load_v<kV>(w2 + j), s2v);
    a = mac_vv(a, x, aie::load_v<kV>(w2 + kTile + j));
    // silu(a) = a * sigmoid(a), fp32 throughout (see vsigmoid32)
    const v32f af = a.template to_vector<float>();
    aie::store_v(dst + j, fmul32(af, vsigmoid32(af)));
    // state shift
    aie::store_v(ns0 + j, s1v);
    aie::store_v(ns1 + j, s2v);
    v32b xh, xl;
    split32(x, xh, xl);
    aie::store_v(ns2 + j, xh);
  }
  if (t < 4) {
    // L2-normalise the 8 heads of this tile in place
#pragma clang loop unroll(disable)
    for (unsigned hh = 0; hh < kTile / kHD; ++hh) {
      float *__restrict hp = dst + hh * kHD;
      accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
      for (unsigned j = 0; j < kHD; j += kV) {
        v32b ch, cl;
        split32(aie::load_v<kV>(hp + j), ch, cl);
        ss = aie::mac(ss, ch, ch);
        ss = aie::mac(ss, ch, cl);
        ss = aie::mac(ss, ch, cl);
      }
      // aie::invsqrt is a coarse hardware approximation (~2% observed);
      // two Newton steps bring it to fp32.
      const float x = aie::reduce_add(ss.template to_vector<float>()) + 1e-6f;
      float inv = aie::invsqrt(x);
      inv = inv * (1.5f - 0.5f * x * inv * inv);
      inv = inv * (1.5f - 0.5f * x * inv * inv);
      const bfloat16 ih = (bfloat16)inv;
      const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
      for (unsigned j = 0; j < kHD; j += kV) {
        accf32 o = aie::zeros<accfloat, kV>();
        o = mac_vs(o, aie::load_v<kV>(hp + j), ih, il);
        aie::store_v(hp + j, o.template to_vector<float>());
      }
    }
  }
}

// ---- emit record for head h (v tile already in vt): rec fp32[512]
static inline void glue_emit(const float *__restrict qk, const float *__restrict vt,
                             const float *__restrict decay, const float *__restrict beta,
                             float *__restrict rec, unsigned h) {
  const unsigned kh = h / 2;
  const float *kp = qk + 2048 + kh * kHD;
  const float *qp = qk + kh * kHD;
  const float *vp = vt + (h % 8) * kHD;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHD; j += kV) {
    aie::store_v(rec + j, aie::load_v<kV>(kp + j));
    aie::store_v(rec + kHD + j, aie::load_v<kV>(qp + j));
    aie::store_v(rec + 2 * kHD + j, aie::load_v<kV>(vp + j));
    aie::store_v(rec + 3 * kHD + j, aie::zeros<float, kV>());
  }
  rec[384] = decay[h];
  rec[385] = beta[h];
}
