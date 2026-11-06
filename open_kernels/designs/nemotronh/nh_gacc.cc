// Nemotron-H nh_moe: a Q4_0-native down band of slot e into the act scratch at ms + 128, and, after the band's
// last group, onto band `band` of the core's accumulator (fresh for e == 0). gemv_q4_tile writes the band in row
// order only at its last k-tile, so the sum over the slots is taken here, one band at a time (gm_gacc's form).
#define GEMV_PER_CALL 2
#define GEMV_Q4_0_NATIVE 1
#include "gemv_q4.h"
#include "vecmath.h"

extern "C" {
void nh_gacc(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
             float *__restrict acc, int32_t group, int32_t per_band, int32_t band, int32_t e, int32_t ngroups) {
  float *__restrict y = ms + 128;
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, 2);
  if (group != ngroups - 1)
    return;
  float *__restrict a = acc + band * 64;
  aie::set_rounding(aie::rounding_mode::conv_even);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; j += 16) {
    const vfN<16> v = aie::load_v<16>(y + j);
    if (e == 0)
      aie::store_v(a + j, v);
    else
      aie::store_v(a + j, faddN<16>(aie::load_v<16>(a + j), v));
  }
}
}
