// Gemma 4 audio SiLU and GLU activation chunks, BF16 in/out.
#include "vecmath.h"
#include <cstdint>

static inline uint16_t bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

extern "C" void gemma4_audio_act_chunk(const bfloat16 *__restrict x,
                                        bfloat16 *__restrict y) {
  uint16_t *bits = reinterpret_cast<uint16_t *>(y);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 256; ++i) {
#ifdef GEMMA4_AUDIO_GLU
    const float a = (float)x[i];
    const float gate = (float)x[256 + i];
    bits[i] = bf16_rne(a / (1.f + sexp(-gate)));
#else
    const float value = (float)x[i];
    bits[i] = bf16_rne(value / (1.f + sexp(-value)));
#endif
  }
}
