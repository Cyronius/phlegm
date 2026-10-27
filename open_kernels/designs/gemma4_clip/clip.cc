// Gemma 4 clippable linear: BF16 input clamp or FP32 GEMM output -> BF16 clamp.
#include <cstdint>

#ifndef WIDTH
#define WIDTH 1024
#endif
#ifndef POST_LINEAR
#define POST_LINEAR 0
#endif

static inline float bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bits.f;
}

#if POST_LINEAR
using Input = float;
#else
using Input = bfloat16;
#endif

extern "C" void gemma4_clip_row(const Input *__restrict input,
                                 const float *__restrict bounds,
                                 bfloat16 *__restrict output) {
  const float low = bounds[0];
  const float high = bounds[1];
#pragma clang loop unroll(disable)
  for (unsigned channel = 0; channel < WIDTH; ++channel) {
#if POST_LINEAR
    float value = bf16_rne(input[channel]);
#else
    float value = (float)input[channel];
#endif
    if (value < low) value = low;
    if (value > high) value = high;
    output[channel] = bfloat16(bf16_rne(value));
  }
}
