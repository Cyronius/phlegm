// Nemotron-H nh_moe: h band = w[e] * relu(u)^2 for one 64-row band of slot e's up projection (ms: u @0) -> one
// f32 y element. The experts are non-gated (transformers NemotronHExperts, llama.cpp LLM_FFN_RELU_SQR); the
// routing weight multiplies h, so the down projection's sum over the slots is already weighted (the shared
// expert's two slots carry 1).
#include "vecmath.h"

extern "C" {
void nh_act(const float *__restrict ms, float *__restrict h, const float *__restrict wv, int32_t e) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const v32f w = aie::broadcast<float, 32>(wv[e]);
  const v32f z = aie::zeros<float, 32>();
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {
    const v32f r = aie::max(aie::load_v<32>(ms + j), z);
    aie::store_v(h + j, fmul32(fmul32(r, r), w));
  }
}
}
