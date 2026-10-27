// Gemma 4 audio subsampler: bias-free LayerNorm over channels, then ReLU.
#include "vecmath.h"
#include <cstdint>

#ifndef CHANNELS
#error CHANNELS must be defined
#endif

static inline uint16_t bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

extern "C" void gemma4_audio_subnorm(const uint8_t *__restrict packed,
                                      bfloat16 *__restrict out) {
  const float *x = reinterpret_cast<const float *>(packed);
  const bfloat16 *gamma = reinterpret_cast<const bfloat16 *>(packed + CHANNELS * 4);
  float sum = 0.f;
  float square = 0.f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < CHANNELS; ++i) {
    sum += x[i];
    square += x[i] * x[i];
  }
  const float mean = sum * (1.f / CHANNELS);
  const float variance = square * (1.f / CHANNELS) - mean * mean;
  const float inv = srsqrt((variance > 0.f ? variance : 0.f) + 1e-6f);
  uint16_t *bits = reinterpret_cast<uint16_t *>(out);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < CHANNELS; ++i) {
    const float value = (x[i] - mean) * inv * (float)gamma[i];
    bits[i] = bf16_rne(value > 0.f ? value : 0.f);
  }
}
