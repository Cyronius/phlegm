// Xing xg: h band = w[e] * silu(g) * u for one 64-row band of slot e (ms: u @0, g @64) -> one f32 y element.
// The experts, the shared expert and the dense MLP's slices are SiLU-gated (transformers Xing4_0MLP, llama.cpp
// LLM_FFN_SILU / LLM_FFN_PAR); silu(x) = x sigmoid(x) as the dense dx and hy_act compute it (vsiluN), then the
// slot's routing weight (1 for the shared expert and the dense slices), so the down projection's sum over the
// slots is already weighted.
#include "vecmath.h"

extern "C" {
void xg_act(const float *__restrict ms, float *__restrict h, const float *__restrict wv, int32_t e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + 0;
  const float *__restrict g = ms + 64;
  const v32f w = aie::broadcast<float, 32>(wv[e]);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 64; i += 32) {
    const v32f a = vsiluN<32>(aie::load_v<32>(g + i));
    aie::store_v(h + i, fmul32(fmul32(a, aie::load_v<32>(u + i)), w));
  }
}
}
