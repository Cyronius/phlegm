#pragma once
// MoE combine (decode): out = xres + sum_e w[e] * y_e + sigmoid(xm . sgw) * shared
//   w[8]   from the router output element (fp32 at float index 264..271)
//   y_e    routed expert e's down-projection output, fp32[2048] (two 4 KB elements)
//   shared shared-expert down output fp32[2048]; xm bf16[2048] (MoE input); sgw bf16[2048]
// Reference: decode_step.py moe_decode.
#include "vecmath.h"

static constexpr unsigned kN = 2048;
static constexpr unsigned kV = 32;

static inline void mc_wcopy_impl(const float *__restrict rout, float *__restrict w) {
  for (unsigned k = 0; k < 8; ++k) w[k] = rout[264 + k];
}

// acc = (e == 0 ? 0 : acc) + w[e] * y
static inline void mc_axpy_impl(float *__restrict acc, const float *__restrict y0, const float *__restrict y1,
                                const float *__restrict w, int e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float we = w[e];
  const bfloat16 wh = (bfloat16)we;
  const bfloat16 wl = (bfloat16)(we - (float)wh);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *yp = (j < 1024) ? (y0 + j) : (y1 + (j - 1024));
    accf32 a;
    if (e == 0)
      a = aie::zeros<accfloat, kV>();
    else
      a.from_vector(aie::load_v<kV>(acc + j));
    a = mac_vs(a, aie::load_v<kV>(yp), wh, wl);
    aie::store_v(acc + j, a.template to_vector<float>());
  }
}

static inline void mc_fin_impl(const float *__restrict acc, const float *__restrict x0, const float *__restrict x1,
                               const float *__restrict s0, const float *__restrict s1,
                               const bfloat16 *__restrict xm, const bfloat16 *__restrict sgw,
                               float *__restrict o0, float *__restrict o1) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 d = aie::zeros<accfloat, kV>();
  for (unsigned j = 0; j < kN; j += kV)
    d = aie::mac(d, aie::load_v<kV>(xm + j), aie::load_v<kV>(sgw + j));
  const float sg = ssigmoid(aie::reduce_add(d.template to_vector<float>()));
  const bfloat16 gh = (bfloat16)sg;
  const bfloat16 gl = (bfloat16)(sg - (float)gh);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kN; j += kV) {
    const float *xp = (j < 1024) ? (x0 + j) : (x1 + (j - 1024));
    const float *sp = (j < 1024) ? (s0 + j) : (s1 + (j - 1024));
    float *op = (j < 1024) ? (o0 + j) : (o1 + (j - 1024));
    accf32 a;
    a.from_vector(fadd32(aie::load_v<kV>(xp), aie::load_v<kV>(acc + j)));
    a = mac_vs(a, aie::load_v<kV>(sp), gh, gl);
    aie::store_v(op, a.template to_vector<float>());
  }
}
