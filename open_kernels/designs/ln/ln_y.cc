// One entry point per TU (IRON compiles a source once per ExternalFunction); the math is ln.h.
#include "ln.h"

extern "C" {
// y half i = x_i + a_i (one output element)
void ln_y(const float *__restrict x0, const float *__restrict x1, const float *__restrict a0,
          const float *__restrict a1, float *__restrict y, int32_t i) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *x = i ? x1 : x0;
  const float *a = i ? a1 : a0;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < kHalf; j += kV) aie::store_v(y + j, fadd32(aie::load_v<kV>(x + j), aie::load_v<kV>(a + j)));
}
}
