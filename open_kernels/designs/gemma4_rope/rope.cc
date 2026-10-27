// Half-split BF16 Q/K rotation for Gemma 4 local or full attention.
#include <cstdint>

#ifndef HEAD_DIM
#error HEAD_DIM must be defined by the IRON specialization
#endif

static inline uint16_t to_bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

extern "C" void gemma4_rope_row(const bfloat16 *__restrict x,
                                 const float *__restrict table,
                                 bfloat16 *__restrict y) {
  uint16_t *out = reinterpret_cast<uint16_t *>(y);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < HEAD_DIM / 2; ++i) {
    const float a = (float)x[i];
    const float b = (float)x[i + HEAD_DIM / 2];
    const float c = table[i];
    const float s = table[i + HEAD_DIM / 2];
    out[i] = to_bf16_rne(a * c - b * s);
    out[i + HEAD_DIM / 2] = to_bf16_rne(b * c + a * s);
  }
}
