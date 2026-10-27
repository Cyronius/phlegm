// Four GPT-OSS expert outputs are in router-ID order, 64 rows per block.
#include <cstdint>

extern "C" void gptoss_mix(const float *__restrict experts,
                            const uint8_t *__restrict route,
                            float *__restrict result) {
  float weights[4];
  __builtin_memcpy(weights, route + 16, sizeof(weights));
#pragma clang loop unroll(disable)
  for (unsigned row = 0; row < 64; ++row) {
    float sum = 0.0f;
    for (unsigned expert = 0; expert < 4; ++expert)
      sum += weights[expert] * experts[expert * 64 + row];
    result[row] = sum;
  }
}
