// Half-split RoPE for one 64-value BF16 Q or K head. The supplied table
// includes YaRN's frequency interpolation and attention factor.
#include <cstdint>

static inline uint16_t to_bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

extern "C" void gptoss_rope_row(const bfloat16 *__restrict x,
                                const float *__restrict table,
                                bfloat16 *__restrict y) {
  uint16_t *out = reinterpret_cast<uint16_t *>(y);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 32; ++i) {
    const float a = (float)x[i];
    const float b = (float)x[i + 32];
    const float c = table[i];
    const float s = table[i + 32];
    out[i] = to_bf16_rne(a * c - b * s);
    out[i + 32] = to_bf16_rne(b * c + a * s);
  }
}
