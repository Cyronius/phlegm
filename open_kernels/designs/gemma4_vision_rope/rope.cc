// Axial 2D RoPE: lanes 0..31 use axis 0; lanes 32..63 use axis 1.
#include <cstdint>

static inline uint16_t bf16(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return (uint16_t)(bits.u >> 16);
}

extern "C" void gemma4_vision_rope_row(const bfloat16 *__restrict x,
                                       const float *__restrict table,
                                       bfloat16 *__restrict y) {
  uint16_t *out = reinterpret_cast<uint16_t *>(y);
#pragma clang loop unroll(disable)
  for (unsigned axis = 0; axis < 2; ++axis) {
    const unsigned off = axis * 32;
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < 16; ++i) {
      const float a = (float)x[off + i];
      const float b = (float)x[off + i + 16];
      const float c = table[off + i];
      const float s = table[off + i + 16];
      out[off + i] = bf16(a * c - b * s);
      out[off + i + 16] = bf16(b * c + a * s);
    }
  }
}
