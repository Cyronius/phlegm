// One 32x128 GPT-OSS half-width q4_1 chunk. The source is the ordinary
// half-chunk raster, selected from the container's four-row supertile.
#include "vecmath.h"
#include <cstdint>

extern "C" void gptoss_qkv_step(const uint8_t *__restrict chunk,
                                 const bfloat16 *__restrict x_bias,
                                 float *__restrict acc, int32_t band, int32_t kb) {
  const bfloat16 *meta = reinterpret_cast<const bfloat16 *>(chunk);
  for (unsigned r = 0; r < 32; ++r) {
    float sum = kb == 0 ? 0.f : acc[r];
#pragma clang loop unroll(disable)
    for (unsigned c = 0; c < 128; ++c) {
      const unsigned block = c >> 5;
      const unsigned slot = block * 32 + r;
      const float scale = (float)meta[slot];
      const float minv = (float)meta[128 + slot];
      const unsigned at = 512 + (r >> 4) * 1024 + block * 256
                        + (c & 31) * 8 + ((r & 15) >> 1);
      const uint8_t packed = chunk[at];
      const unsigned code = (r & 1) ? packed >> 4 : packed & 15;
      sum += (scale * (float)code + minv) * (float)x_bias[(unsigned)kb * 128 + c];
    }
    if (kb == 22) sum += (float)x_bias[2944 + (unsigned)band * 32 + r];
    acc[r] = sum;
  }
}
