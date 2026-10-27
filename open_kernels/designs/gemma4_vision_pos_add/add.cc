// Gemma 4 vision patch projection plus learned 2D position, with BF16 rounds.
#include <cstdint>

static inline float bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bits.f;
}

extern "C" void gemma4_vision_pos_add_row(const bfloat16 *__restrict projected,
                                           const bfloat16 *__restrict positions,
                                           bfloat16 *__restrict output) {
#pragma clang loop unroll(disable)
  for (unsigned channel = 0; channel < 768; ++channel) {
    const float position = bf16_rne((float)positions[channel] +
                                    (float)positions[768 + channel]);
    const float result = bf16_rne((float)projected[channel] + position);
    output[channel] = bfloat16(result);
  }
}
