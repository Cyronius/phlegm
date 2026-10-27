// Gemma 4 E2B/E4B per-layer input: GELU-tanh(proj) * layer_input.
#include "vecmath.h"
#include <cstdint>

static inline uint16_t to_bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

static inline float tanh_scalar(float x) {
  const float a = x < 0.0f ? -x : x;
  if (a > 20.0f) return x < 0.0f ? -1.0f : 1.0f;
  const float e = sexp(-2.0f * a);
  const float t = (1.0f - e) / (1.0f + e);
  return x < 0.0f ? -t : t;
}

extern "C" void gemma4_ple_gate(const bfloat16 *__restrict projection,
                                 const bfloat16 *__restrict layer_input,
                                 bfloat16 *__restrict gated) {
  uint16_t *out = reinterpret_cast<uint16_t *>(gated);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 256; ++i) {
    const float x = (float)projection[i];
    const float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    const float activation = 0.5f * x * (1.0f + tanh_scalar(z));
    union { uint32_t u; float f; } rounded = {uint32_t(to_bf16_rne(activation)) << 16};
    out[i] = to_bf16_rne(rounded.f * (float)layer_input[i]);
  }
}
