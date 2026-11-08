// Xing xg: element i of a bf16 activation of K values (2048 per 4 KB element) into the GEMV table: blocks
// [64 i, min(64 i + 64, K/32)) (gm_prep's arithmetic, nh_prep's form).
#define GEMV_Q4_0_NATIVE 1
#include "gemv_q4.h"

extern "C" {
void xg_prep(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 64u * (unsigned)i;
  const unsigned nb = (b0 + 64u <= total) ? 64u : total - b0;
  gemv_q4_prep_blocks(e, tab, (unsigned)K, b0, nb);
}
}
