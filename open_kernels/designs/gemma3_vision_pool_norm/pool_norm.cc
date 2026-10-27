// Gemma 3 projector: BF16 AvgPool2d(4), then RMSNorm with (1 + weight).
#include "vecmath.h"
#include <cstdint>

static inline float bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bits.f;
}

extern "C" void gemma3_vision_pool_norm_group(const bfloat16 *__restrict patches,
                                                const bfloat16 *__restrict weight,
                                                bfloat16 *__restrict output) {
  alignas(64) static float pooled[1152];
  float square_sum = 0.f;
#pragma clang loop unroll(disable)
  for (unsigned channel = 0; channel < 1152; ++channel) {
    float total = 0.f;
#pragma clang loop unroll(disable)
    for (unsigned patch = 0; patch < 16; ++patch)
      total += (float)patches[patch * 1152 + channel];
    const float value = bf16_rne(total * (1.f / 16.f));
    pooled[channel] = value;
    square_sum += value * value;
  }
  const float inv = srsqrt(square_sum * (1.f / 1152.f) + 1e-6f);
#pragma clang loop unroll(disable)
  for (unsigned channel = 0; channel < 1152; ++channel) {
    const float scale = 1.f + (float)weight[channel];
    output[channel] = bfloat16(bf16_rne(pooled[channel] * inv * scale));
  }
}
