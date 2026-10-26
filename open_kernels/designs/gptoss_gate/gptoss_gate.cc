// GPT-OSS expert activation for 64 rows. Inputs include projection biases.
// h = bf16((clip(up,-7,7)+1) * min(gate,7) * sigmoid(1.702*min(gate,7))).
#include "vecmath.h"

extern "C" void gptoss_gate(const float *__restrict gate,
                             const float *__restrict up,
                             bfloat16 *__restrict hidden) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const v32f seven = aie::broadcast<float, 32>(7.0f);
  const v32f minus_seven = aie::broadcast<float, 32>(-7.0f);
  const v32f one = aie::broadcast<float, 32>(1.0f);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 32) {
    const v32f g = aie::min(aie::load_v<32>(gate + j), seven);
    const v32f u = aie::max(aie::min(aie::load_v<32>(up + j), seven), minus_seven);
    const v32f sigmoid = vsigmoidN<32>(fscaleN<32>(g, 1.702f));
    const v32f y = fmulN<32>(faddN<32>(u, one), fmulN<32>(g, sigmoid));
    accf32 a;
    a.from_vector(y);
    aie::store_v(hidden + j, a.template to_vector<bfloat16>());
  }
}
