// down: k-tile ky of a band into the core's y accumulator
#include "xmb.h"
extern "C" {
void xmb_step_dn(const uint8_t *__restrict band, const bfloat16 *__restrict hb, float *__restrict c, int tile, int ky) {
  xmb_step_tile(band, ky, hb + tile * 64 * XMB_NT, c);
}
}
