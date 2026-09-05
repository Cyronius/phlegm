// The same from an fp32 activation (rounded to bf16 first): blocks b0 .. b0+nb-1 of a K-long
// activation, 1024 floats per 4 KB element -> 32 blocks per element.
#include "gemv_q4.h"

extern "C" {
void gemv_q4_prep_f32_rt(const float *__restrict xf, uint8_t *__restrict tab, int32_t K, int32_t b0, int32_t nb) {
  gemv_q4_prep_f32_blocks(xf, tab, (unsigned)K, (unsigned)b0, (unsigned)nb);
}
}
