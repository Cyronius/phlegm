// GPT-OSS epsilon is 1e-5. Round the BF16 projection input to nearest-even.
#include "vecmath.h"
#include <cstdint>

static inline uint16_t to_bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

extern "C" void gptoss_norm(const float *__restrict x,
                             const bfloat16 *__restrict weight,
                             bfloat16 *__restrict out) {
  float sum = 0.f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 2880; ++i) sum += x[i] * x[i];
  const float inv = srsqrt(sum * (1.f / 2880.f) + 1e-5f);
  uint16_t *bits = reinterpret_cast<uint16_t *>(out);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 2880; ++i)
    bits[i] = to_bf16_rne((x[i] * inv) * (float)weight[i]);
}
