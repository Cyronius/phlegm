// Layer RMSNorm without a residual add, emitting fp32 half i (Gemma's post-attention / post-FFN
// norms: t = norm(out) * w, added to the residual by the next ln call):
//   y_i = (x * rsqrt(mean(x^2) + eps) * w)[half i]
// Elements are LN_N*2 bytes: x as two fp32 halves, w bf16[LN_N], y_i one fp32 half. One entry per TU.
#include "ln.h"

extern "C" {
void ln_nr32(const float *__restrict x0, const float *__restrict x1, const bfloat16 *__restrict w,
             float *__restrict y, int32_t i) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 ss = aie::zeros<accfloat, kV>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < kHalf) ? (x0 + j) : (x1 + (j - kHalf));
    v32b h, l;
    split32(aie::load_v<kV>(xp), h, l);
    ss = aie::mac(ss, h, h);
    ss = aie::mac(ss, h, l);
    ss = aie::mac(ss, h, l);
  }
  const float inv = srsqrt(aie::reduce_add(ss.template to_vector<float>()) * (1.0f / kN) + LN_EPS);
  const bfloat16 ih = (bfloat16)inv;
  const bfloat16 il = (bfloat16)(inv - (float)ih);
  const float *x = i ? x1 : x0;
  const bfloat16 *wh = w + (i ? kHalf : 0);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) {
    accf32 t = aie::zeros<accfloat, kV>();
    t = mac_vv(t, aie::load_v<kV>(x + j), aie::load_v<kV>(wh + j));      // x * w  (fp32)
    accf32 o = aie::zeros<accfloat, kV>();
    o = mac_vs(o, t.template to_vector<float>(), ih, il);              // * inv
    aie::store_v(y + j, o.template to_vector<float>());
  }
}
}
