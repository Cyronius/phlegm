// Block-quantise blocks b0 .. b0+nb-1 of a K-long bf16 activation into the GEMV table, every
// parameter at runtime: one entry point serves an activation that arrives in several 4 KB
// elements (K = 2560 is 1.25 elements) and every K of a design (16 KB program memory).
#include "gemv_q4.h"

extern "C" {
void gemv_q4_prep_rt(const bfloat16 *__restrict x, uint8_t *__restrict tab, int32_t K, int32_t b0, int32_t nb) {
  gemv_q4_prep_blocks(x, tab, (unsigned)K, (unsigned)b0, (unsigned)nb);
}
}
