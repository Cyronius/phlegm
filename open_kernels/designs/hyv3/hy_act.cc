// h band = w[2 j + half] * silu(g) * u for one 64-row band (ms: u @0, g @64) -> one f32 y
// element: the core's j-th gate | up slot under the pair law (recipes/hyv3.py core_slots). silu(x) =
// x sigmoid(x) as the dense dx computes it (vsiluN), then the slot's routing weight (1 for the shared
// expert and the dense MLP's slices).
#include "vecmath.h"

extern "C" {
void hy_act(const float *__restrict ms, float *__restrict h, const float *__restrict wv, int32_t j, int32_t half) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ms + 0;
  const float *__restrict g = ms + 64;
  const v32f w = aie::broadcast<float, 32>(wv[2 * j + half]);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 64; i += 32) {
    const v32f a = vsiluN<32>(aie::load_v<32>(g + i));
    aie::store_v(h + i, fmul32(fmul32(a, aie::load_v<32>(u + i)), w));
  }
}
}
