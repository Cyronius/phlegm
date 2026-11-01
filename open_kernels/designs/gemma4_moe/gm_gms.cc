// Gemma 4 26B-A4B gm_moe (designs/gemma4_moe/gm.py): a Q4_0-native 64-row band of an expert's gate or
// up into the act scratch at ms + dst (gate at 64, up at 0), band law rs = 2 (std_perm's 64-row bands).
#define GEMV_PER_CALL 2
#define GEMV_Q4_0_NATIVE 1
#include "gemv_q4.h"

extern "C" {
void gm_gms(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict ms,
            int32_t group, int32_t per_band, int32_t dst) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, ms + dst, (unsigned)per_band, 2);
}
}
