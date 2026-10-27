// SigLIP Conv2d bias, BF16 activation boundary, then learned position add.
#include <cstdint>

static inline uint16_t bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  return uint16_t(bits.u >> 16);
}

extern "C" void gemma3_vision_patch_finish_row(const float *__restrict projected,
                                                 const bfloat16 *__restrict addends,
                                                 bfloat16 *__restrict output) {
  uint16_t *bits = reinterpret_cast<uint16_t *>(output);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 1152; ++i) {
    const uint16_t patch_bits = bf16_rne(projected[i] + float(addends[i]));
    union { uint32_t u; float f; } patch = {uint32_t(patch_bits) << 16};
    bits[i] = bf16_rne(patch.f + float(addends[1152 + i]));
  }
}
