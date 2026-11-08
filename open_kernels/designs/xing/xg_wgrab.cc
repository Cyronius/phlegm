// Xing xg: the slots' weights (f32[16], the host's xg_pre: the top 4 then 1 for the shared expert, or the dense
// MLP's nine slices at 1) from xm's last x element, byte XG_W_OFF (recipes/xing.py layout W_OFF: past the
// hidden bf16 values).
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef XG_W_OFF
#error "XG_W_OFF (recipes/xing.py layout W_OFF) is a compile-time flag"
#endif

extern "C" {
void xg_wgrab(const bfloat16 *__restrict xe, float *__restrict wv) {
  aie::store_v(wv, aie::load_v<16>((const float *)((const uint8_t *)xe + XG_W_OFF)));
}
}
