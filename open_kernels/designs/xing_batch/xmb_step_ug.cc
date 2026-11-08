// up / gate: k-tile ky of a band into band `half` of the core's u / g buffer
#include "xmb.h"
extern "C" {
void xmb_step_ug(const uint8_t *__restrict band, const bfloat16 *__restrict xa, float *__restrict c, int ky, int half) {
  xmb_step_tile(band, ky, xa, c + half * 64 * XMB_NT);
}
}
