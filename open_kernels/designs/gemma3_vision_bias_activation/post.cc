// BF16 linear bias boundary; FC1 also applies SigLIP's tanh GELU.
#include "vecmath.h"
#include <cstdint>

static inline float bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bits.f;
}

static inline float gelu_tanh(float x) {
  const float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
  const float a = z < 0.f ? -z : z;
  const float e = sexp(-2.f * a);
  const float t = (z < 0.f ? -1.f : 1.f) * (1.f - e) / (1.f + e);
  return 0.5f * x * (1.f + t);
}

extern "C" void gemma3_vision_bias_activation_row(const float *__restrict projected,
                                                    const bfloat16 *__restrict bias,
                                                    bfloat16 *__restrict output) {
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < LOGICAL; ++i) {
    float value = bf16_rne(projected[i] + float(bias[i]));
#if GELU
    value = gelu_tanh(value);
#endif
    output[i] = bfloat16(bf16_rne(value));
  }
#pragma clang loop unroll(disable)
  for (unsigned i = LOGICAL; i < PADDED; ++i)
    output[i] = bfloat16(0.f);
}
