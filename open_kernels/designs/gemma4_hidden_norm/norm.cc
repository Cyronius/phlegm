// BF16 RMSNorm with epsilon 1e-6, then BF16 residual addition.
#include "vecmath.h"
#include <cstdint>

#ifndef HIDDEN
#error HIDDEN must be defined by the IRON specialization
#endif

static inline uint16_t to_bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

extern "C" void gemma4_hidden_norm(const bfloat16 *__restrict pair,
                                    const bfloat16 *__restrict scale,
                                    bfloat16 *__restrict output) {
  float sum = 0.0f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < HIDDEN; ++i) {
    const float x = (float)pair[i];
    sum += x * x;
  }
  const float inv = srsqrt(sum * (1.0f / HIDDEN) + 1e-6f);
  uint16_t *bits = reinterpret_cast<uint16_t *>(output);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < HIDDEN; ++i) {
    const uint16_t normalized_bits = to_bf16_rne(((float)pair[i] * inv) * (float)scale[i]);
    union { uint32_t u; float f; } normalized = {uint32_t(normalized_bits) << 16};
    bits[i] = to_bf16_rne((float)pair[HIDDEN + i] + normalized.f);
  }
}
