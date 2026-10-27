// SigLIP LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta.
#include "vecmath.h"
#include <cstdint>

static inline float bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bits.f;
}

extern "C" void gemma3_vision_layernorm_row(const bfloat16 *__restrict input,
                                              const bfloat16 *__restrict params,
                                              bfloat16 *__restrict output) {
  float sum = 0.f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 1152; ++i)
    sum += float(input[i]);
  const float mean = sum * (1.f / 1152.f);
  float square_sum = 0.f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 1152; ++i) {
    const float delta = float(input[i]) - mean;
    square_sum += delta * delta;
  }
  const float inv = srsqrt(square_sum * (1.f / 1152.f) + 1e-6f);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 1152; ++i) {
    const float y = (float(input[i]) - mean) * inv * float(params[i]) +
                    float(params[1152 + i]);
    output[i] = bfloat16(bf16_rne(y));
  }
}
