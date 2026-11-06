// Nemotron-H nh_moe: element i of an fp32 activation of K values (1024 per 4 KB element; the fifo types it as
// bf16) into the GEMV table: blocks [32 i, min(32 i + 32, K/32)) (gm_prepf's arithmetic).
#define GEMV_Q4_0_NATIVE 1
#include "gemv_q4.h"

extern "C" {
void nh_prepf(const bfloat16 *__restrict e, uint8_t *__restrict tab, int32_t K, int32_t i) {
  const unsigned total = (unsigned)K / 32, b0 = 32u * (unsigned)i;
  const unsigned nb = (b0 + 32u <= total) ? 32u : total - b0;
  gemv_q4_prep_f32_blocks((const float *)e, tab, (unsigned)K, b0, nb);
}
}
