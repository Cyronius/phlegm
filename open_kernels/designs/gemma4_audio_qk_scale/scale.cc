// Gemma 4 audio Q/K scaling after BF16 clippable projections.
#include "vecmath.h"
#include <cstdint>

static inline float bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bits.f;
}

extern "C" void gemma4_audio_qk_scale_row(const bfloat16 *__restrict input,
                                            const bfloat16 *__restrict learned,
                                            float *__restrict output,
                                            const int row) {
  const bool query = (row % 36) < 12;
  const float q_scale = 0.127517431f; // (128^-0.5) / log(2)
  const float k_scale = 1.894636124f; // log(1+e) / log(2)
#pragma clang loop unroll(disable)
  for (unsigned channel = 0; channel < 128; ++channel) {
    const float value = (float)input[channel];
    if (query) {
      const float softplus = bf16_rne(ssoftplus((float)learned[channel]));
      output[channel] = (value * q_scale) * softplus;
    } else {
      output[channel] = value * k_scale;
    }
  }
}
