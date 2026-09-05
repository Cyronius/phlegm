// One entry point per TU (IRON compiles a source once per ExternalFunction); the math is ln.h.
#include "ln.h"

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
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kN) + LN_EPS);
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
