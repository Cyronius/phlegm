// One GPT-OSS MXFP4 container chunk: 32 rows x 128 columns, fp32 output.
// Layout and E8M0 handling follow open_kernels/model/mxfp4.py.
#include <cstdint>

extern "C" void gptoss_mxfp4(const uint8_t *__restrict chunk,
                              float *__restrict values) {
  static constexpr int8_t level[16] =
      {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  for (unsigned r = 0; r < 32; ++r) {
    const unsigned rg = r >> 4;
    const unsigned pair_row = (r & 15) >> 1;
    const unsigned high = r & 1;
    for (unsigned c = 0; c < 128; ++c) {
      const uint8_t exponent = chunk[(c >> 5) * 32 + r];
      const uint32_t bits = exponent < 2 ? (0x00200000u << exponent)
                                           : ((uint32_t)(exponent - 1) << 23);
      union { uint32_t u; float f; } scale = {bits};
      const unsigned byte_index = 512 + (rg * 64 + (c >> 1)) * 16
                                + (c & 1) * 8 + pair_row;
      const uint8_t packed = chunk[byte_index];
      const uint8_t code = high ? (packed >> 4) : (packed & 15);
      values[r * 128 + c] = (float)level[code] * scale.f;
    }
  }
}
