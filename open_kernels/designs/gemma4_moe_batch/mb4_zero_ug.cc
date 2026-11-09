#include "mb4.h"
extern "C" {
// Every gate / up pass starts here, before its first dequantised weight is rounded to bf16 (mb_step_parity's
// to_vector), so the rounding mode is set here and not only in mb_gelu: a core comes up in its default mode
// after a context change, and the first bands of that dispatch were rounded in it (xing_batch/xmb_zero_ug.cc;
// open_kernels/UPSTREAM.md, 2026-11-08).
void mb_zero_ug(float *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  mb_zero_n(c, MB_NRB_UP * 64 * MB_NT);
}
}
