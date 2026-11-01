// Gemma 4 26B-A4B gm8 (designs/gemma4_moe/gm8.py): h band = w[e] * gelu_tanh(g) * u for one 64-row band
// (ms: u @0, g @64) into band `band` of the core's own f32 h (an x-typed L1 buffer, 768 floats). The
// arithmetic is gm_act.cc's (dense_act's gelu_tanh, then the routing weight); only the destination moves.
#include "vecmath.h"

extern "C" {
void gm8_act(const float *__restrict ms, bfloat16 *__restrict hb, const float *__restrict wv, int32_t e,
             int32_t band) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + 0;
  const float *__restrict g = ms + 64;
  float *__restrict h = (float *)hb + band * 64;
  const v32f w = aie::broadcast<float, 32>(wv[e]);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {
    const v32f x = aie::load_v<32>(g + j);
    const v32f x3 = fmul32(fmul32(x, x), x);
    const v32f z2 = fscaleN<32>(fadd32(x, fscaleN<32>(x3, 0.044715f)), 2.0f * 0.7978845608028654f);
    const v32f a = fmul32(x, vsigmoidN<32>(z2));
    aie::store_v(h + j, fmul32(fmul32(a, aie::load_v<32>(u + j)), w));
  }
}
}
