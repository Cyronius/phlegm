// Layer RMSNorm with fused residual add (decode, one call):
//   y  = x + add                       (fp32 [2048], the new residual)
//   xn = bf16( y * rsqrt(mean(y^2) + 1e-6) * w )
// Elements are 4 KB: x, add, y as two fp32[1024] halves; w, xn as bf16[2048].
#include "vecmath.h"

// LN_N: the width (designs/ln/ln.py and the whole-layer designs pass it from the ModelSpec);
// elements are LN_N*2 bytes: x / add / y as two f32 halves, w / xn as one bf16 element.
#ifndef LN_N
#define LN_N 2048
#endif
static constexpr unsigned kN = LN_N;
static constexpr unsigned kHalf = LN_N / 2;
static constexpr unsigned kV = 32;
static_assert(kHalf % kV == 0, "LN_N must be a multiple of 64");

extern "C" {
void ln_fn(const float *__restrict x0, const float *__restrict x1, const float *__restrict a0,
           const float *__restrict a1, const bfloat16 *__restrict w, float *__restrict y0,
           float *__restrict y1, bfloat16 *__restrict xn) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    const float *ap = (j < kHalf) ? (a0 + j) : (a1 + (j - kHalf));
    float *yp = (j < kHalf) ? (y0 + j) : (y1 + (j - kHalf));
    const v32f y = fadd32(aie::load_v<kV>(xp), aie::load_v<kV>(ap));
    aie::store_v(yp, y);
    v32b h, l;
    split32(y, h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kN) + 1e-6f);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *yp = (j < kHalf) ? (y0 + j) : (y1 + (j - kHalf));
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, aie::load_v<kV>(yp), aie::load_v<kV>(w + j));      // y * w  (fp32)
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), ih, il);            // * inv
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
}
}
