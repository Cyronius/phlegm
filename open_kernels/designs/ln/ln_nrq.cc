// Layer RMSNorm without a residual add, over quarter elements (the quarter norm helper,
// recipes/dense.py norm_quarter): one HALF of
//   xn = bf16( x * rsqrt(mean(x^2) + eps) * w )
// per call. x arrives as four fp32 quarters (LN_N / 4 values, LN_N bytes each), w and xn as the
// bf16 half the call writes (LN_N / 2 values, LN_N bytes). Every element is LN_N bytes, a quarter
// of the split helper's (designs/lin_layer/ln_nr.cc takes x as two halves and w / xn whole), so a
// width whose split helper does not fit the norm core (7168: five 14 KB elements and the stack
// over 64 KB) holds five 7 KB inputs and one output instead.
//
// Bit-identical to ln_nr: the sum of squares walks the same 32-lane vectors in the same order
// (j = 0 .. LN_N), and each output lane is the same two multiplies of its own x and w.
#include "vecmath.h"

#ifndef LN_N
#define LN_N 7168
#endif
#ifndef LN_EPS
#define LN_EPS 1e-6f
#endif
#ifndef LN_GROUPS
#define LN_GROUPS 1
#endif
#ifndef LN_DIV
#define LN_DIV LN_N
#endif
static_assert(LN_GROUPS == 1, "ln_nrq: the quarter helper has no grouped reduction");
static constexpr unsigned kN = LN_N;
static constexpr unsigned kHalf = LN_N / 2;
static constexpr unsigned kQ = LN_N / 4;
static constexpr unsigned kV = 32;
static_assert(kQ % kV == 0, "LN_N must be a multiple of 128 for the quarter helper");

static inline const float *quarter(const float *__restrict x0, const float *__restrict x1,
                                   const float *__restrict x2, const float *__restrict x3, unsigned j) {
  return j < kQ ? x0 + j : j < 2 * kQ ? x1 + (j - kQ) : j < 3 * kQ ? x2 + (j - 2 * kQ) : x3 + (j - 3 * kQ);
}

extern "C" {
void ln_nrq(const bfloat16 *__restrict w, const float *__restrict x0, const float *__restrict x1,
            const float *__restrict x2, const float *__restrict x3, bfloat16 *__restrict xn, int32_t half) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const v32f y = aie::load_v<kV>(quarter(x0, x1, x2, x3, j));
    v32b h, l;
    split32(y, h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / LN_DIV) + LN_EPS);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
  const unsigned j0 = half ? kHalf : 0u;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) {
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, aie::load_v<kV>(quarter(x0, x1, x2, x3, j0 + j)), aie::load_v<kV>(w + j));
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), ih, il);
    aie::store_v(xn + j, o.template to_vector<bfloat16>());
  }
}
}
