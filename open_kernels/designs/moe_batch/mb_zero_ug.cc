#include "moe_batch.h"
extern "C" {
// Every up / gate pass starts here, before its first dequantised weight is rounded to bf16 (mb_step_parity's
// to_vector), so the rounding mode is set here and not only in mb_silu: a core comes up in its default mode
// after a context change, and the first bands of that dispatch were rounded in it (xing_batch/xmb_zero_ug.cc;
// open_kernels/UPSTREAM.md, 2026-11-08).
void mb_zero_ug(float *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  mb_zero_n(c, 128 * MB_NT);
}
}
