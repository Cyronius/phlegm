// Nemotron-H nh_moe: the slots' weights (f32[8], the host's nh_route: six routed, then 1, 1 for the shared
// expert's halves) from xm's last x element, byte 2048 (recipes/nemotronh.py layout AD_W).
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void nh_wgrab(const bfloat16 *__restrict xe, float *__restrict wv) {
  aie::store_v(wv, aie::load_v<8>((const float *)((const uint8_t *)xe + 2048)));
}
}
