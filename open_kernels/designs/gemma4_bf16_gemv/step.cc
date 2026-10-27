// One FLM [256 K, 32 row] BF16 block into a 32-value FP32 accumulator.
#include "vecmath.h"
#include <cstdint>

extern "C" void gemma4_bf16_gemv_step(const bfloat16 *__restrict weight,
                                        const bfloat16 *__restrict activation,
                                        float *__restrict output,
                                        int32_t k_offset) {
  if (k_offset == 0) {
#pragma clang loop unroll(disable)
    for (unsigned row = 0; row < 32; ++row) output[row] = 0.0f;
  }
#pragma clang loop unroll(disable)
  for (unsigned col = 0; col < 256; ++col) {
    const float x = (float)activation[k_offset + col];
#pragma clang loop unroll(disable)
    for (unsigned row = 0; row < 32; ++row)
      output[row] += (float)weight[col * 32 + row] * x;
  }
}
