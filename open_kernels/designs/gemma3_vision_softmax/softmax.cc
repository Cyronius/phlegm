// SigLIP's bidirectional attention softmax: 4096 keys, head dimension 72.
#include "vecmath.h"
#include <cstdint>

static inline float bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bits.f;
}

extern "C" void gemma3_vision_softmax_row(const float *__restrict scores,
                                            bfloat16 *__restrict probabilities) {
  alignas(64) static float scratch[4096];
  const float scale = 0.11785113019775793f; // 1/sqrt(72)
  float maximum = -3.40282347e38f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 4096; ++i) {
    const float s = scores[i] * scale;
    if (s > maximum) maximum = s;
  }
  float total = 0.f;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 4096; ++i) {
    const float e = sexp(scores[i] * scale - maximum);
    scratch[i] = e;
    total += e;
  }
  const float inv = 1.f / total;
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 4096; ++i)
    probabilities[i] = bfloat16(bf16_rne(scratch[i] * inv));
}
