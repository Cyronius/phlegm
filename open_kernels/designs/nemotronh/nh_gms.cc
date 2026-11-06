// Nemotron-H nh_moe (designs/nemotronh/nh.py): one chunk group of a Q4_0-native 64-row band of an expert's up
// projection into the act scratch at ms + dst, band law rs = 2 (std_perm's 64-row bands).
#define GEMV_PER_CALL 2
#define GEMV_Q4_0_NATIVE 1
#include "gemv_q4.h"

extern "C" {
void nh_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms, int32_t group,
            int32_t per_band, int32_t dst) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}
}
