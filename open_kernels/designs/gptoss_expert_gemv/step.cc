// A 32x128 GPT-OSS MXFP4 chunk contributes to one 32-row expert projection.
// Bias lives in bytes 128..191 of column block zero and is added at the end.
#include "vecmath.h"
#include <cstdint>

extern "C" void gptoss_expert_step(const uint8_t *__restrict chunk,
                                    const bfloat16 *__restrict x,
                                    float *__restrict acc,
                                    bfloat16 *__restrict bias, int32_t kb) {
  static constexpr int8_t level[16] =
      {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  const bfloat16 *bias_in = reinterpret_cast<const bfloat16 *>(chunk + 128);
  for (unsigned r = 0; r < 32; ++r) {
    if (kb == 0) bias[r] = bias_in[r];
    float sum = kb == 0 ? 0.f : acc[r];
    const unsigned rg = r >> 4;
    const unsigned pair_row = (r & 15) >> 1;
    const unsigned high = r & 1;
    for (unsigned c = 0; c < 128; ++c) {
      const uint8_t exponent = chunk[(c >> 5) * 32 + r];
      const uint32_t bits = exponent < 2 ? (0x00200000u << exponent)
                                           : ((uint32_t)(exponent - 1) << 23);
      union { uint32_t u; float f; } scale = {bits};
      const unsigned at = 512 + (rg * 64 + (c >> 1)) * 16
                        + (c & 1) * 8 + pair_row;
      const uint8_t packed = chunk[at];
      const uint8_t code = high ? (packed >> 4) : (packed & 15);
      sum += ((float)level[code] * scale.f) * (float)x[(unsigned)kb * 128 + c];
    }
    if (kb == 22) sum += (float)bias[r];
    acc[r] = sum;
  }
}
