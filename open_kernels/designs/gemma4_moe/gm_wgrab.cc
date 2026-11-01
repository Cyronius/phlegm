// Gemma 4 26B-A4B gm_moe: the routing weights (f32[8], the host's gemma4_route) from xm2's last
// x element, byte 2048 (recipes/gemma4.py moe_layout AD_W).
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {
void gm_wgrab(const bfloat16 *__restrict xe, float *__restrict wv) {
  aie::store_v(wv, aie::load_v<8>((const float *)((const uint8_t *)xe + 2048)));
}
}
