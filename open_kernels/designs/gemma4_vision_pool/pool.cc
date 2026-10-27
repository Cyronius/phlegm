// Gemma 4 E2B/E4B vision pool: 9 patch vectors -> one 768-channel token.
#include <cstdint>

static inline float bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bits.f;
}

extern "C" void gemma4_vision_pool_group(const bfloat16 *__restrict patches,
                                          float *__restrict output) {
#pragma clang loop unroll(disable)
  for (unsigned channel = 0; channel < 768; ++channel) {
    float total = 0.f;
#pragma clang loop unroll(disable)
    for (unsigned patch = 0; patch < 9; ++patch)
      total += (float)patches[patch * 768 + channel];
    const float averaged = bf16_rne(total * (1.f / 9.f));
    output[channel] = averaged * 27.71281292f; // sqrt(768)
  }
}
