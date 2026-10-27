// Gemma 4 text Q/K/V per-head RMSNorm, epsilon 1e-6.
#include "vecmath.h"
#include <cstdint>

#ifndef HEAD_DIM
#error HEAD_DIM must be defined by the IRON specialization
#endif

static inline uint16_t to_bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

extern "C" void gemma4_head_norm_row(const bfloat16 *__restrict x,
                                      const bfloat16 *__restrict weight,
                                      bfloat16 *__restrict out) {
  float sum = 0.f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < HEAD_DIM; ++i) {
    const float v = (float)x[i];
    sum += v * v;
  }
  const float inv = srsqrt(sum * (1.f / HEAD_DIM) + 1e-6f);
  uint16_t *bits = reinterpret_cast<uint16_t *>(out);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < HEAD_DIM; ++i)
    bits[i] = to_bf16_rne(((float)x[i] * inv) * (float)weight[i]);
}
