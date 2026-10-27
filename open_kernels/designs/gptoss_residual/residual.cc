// GPT-OSS attention tail: residual + (o_proj + BF16 o_proj bias).
#include "vecmath.h"

extern "C" void gptoss_residual(const float *__restrict pair,
                                 const bfloat16 *__restrict bias,
                                 float *__restrict result) {
#pragma clang loop unroll(disable)
  for (unsigned row = 0; row < 64; ++row) {
    const float projected = pair[64 + row] + (float)bias[row];
    result[row] = pair[row] + projected;
  }
}
