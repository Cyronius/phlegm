// Gemma 4 26B-A4B gm8 (designs/gemma4_moe/gm8.py): one group of a Q4_0-native down band (K = 768) of the
// core's own expert straight into its y element; the band is complete after its last group (gemv_q4.h's
// band law rs = 2, as gm_gms.cc). No sum over experts here: each core's y is one routed slot's partial.
#define GEMV_PER_CALL 2
#define GEMV_Q4_0_NATIVE 1
#include "gemv_q4.h"

extern "C" {
void gm8_dn(const uint8_t *__restrict t, const uint8_t *__restrict tab, float *__restrict y, int32_t group,
            int32_t per_band) {
  gemv_q4_pool_group_rt(t, tab, (unsigned)group, y, (unsigned)per_band, 2);
}
}
