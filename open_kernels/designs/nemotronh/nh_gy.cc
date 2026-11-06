// Nemotron-H (designs/nemotronh/nh.py): one chunk group of a Q4_0-native 64-row band into the band's y element
// (runtime band law: per_band chunks, row split rs). The projection GEMVs' only arithmetic.
#define GEMV_PER_CALL 2
#define GEMV_Q4_0_NATIVE 1
#include "gemv_q4.h"

extern "C" {
void nh_gy(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y, int32_t group,
           int32_t per_band, int32_t rs) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, (unsigned)rs);
}
}
