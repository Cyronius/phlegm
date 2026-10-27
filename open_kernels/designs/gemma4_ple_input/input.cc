// Gemma 4's per-layer context projection scaling, RMSNorm, and optional token mix.
#include "vecmath.h"
#include <cstdint>

#ifndef HIDDEN
#error HIDDEN must be defined
#endif
#ifndef HAS_TOKEN
#error HAS_TOKEN must be defined
#endif

static inline uint16_t to_bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

static inline float from_bits(uint16_t value) {
  union { uint32_t u; float f; } bits = {uint32_t(value) << 16};
  return bits.f;
}

extern "C" void gemma4_ple_input(const bfloat16 *__restrict pair,
                                   const bfloat16 *__restrict scale,
                                   bfloat16 *__restrict output) {
  const float projection_scale = HIDDEN == 1536 ? 0.02551551815399144f : 0.01976423537605237f;
  float context[256];
  float sum = 0.0f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 256; ++i) {
    context[i] = from_bits(to_bf16_rne((float)pair[i] * projection_scale));
    sum += context[i] * context[i];
  }
  const float inv = srsqrt(sum * (1.0f / 256.0f) + 1e-6f);
  uint16_t *bits = reinterpret_cast<uint16_t *>(output);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 256; ++i) {
    uint16_t norm_bits = to_bf16_rne(context[i] * inv * (float)scale[i]);
#if HAS_TOKEN
    const float mixed = from_bits(to_bf16_rne(from_bits(norm_bits) + (float)pair[256 + i]));
    bits[i] = to_bf16_rne(mixed * 0.7071067811865476f);
#else
    bits[i] = norm_bits;
#endif
  }
}
