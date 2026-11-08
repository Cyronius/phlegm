#include "xmb.h"
extern "C" {
// Every gate / up band starts here, before its first dequantised weight is rounded to bf16 (xmb_step_parity's
// to_vector), so the rounding mode is set here and not only in xmb_silu: a core comes up in the default mode, and
// mbl's first band of a dispatch on a fresh core (the first after a context change) was rounded in it -- a different
// h for the first round's slots than every later dispatch computes from the same inputs (2026-11-08 probe:
// h vs fp64 3.16e-3 on the first dispatch, 2.78e-3 after; y consistent with its own h either way).
void xmb_zero_ug(float *__restrict c) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  xmb_zero_n(c, XMB_NRB_UP * 64 * XMB_NT);
}
}
