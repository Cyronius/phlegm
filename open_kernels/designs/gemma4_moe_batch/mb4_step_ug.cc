// up / gate: k-tile ky of a band into band `half` of the core's u / g buffer
#include "mb4.h"
extern "C" {
void mb_step_ug(const uint8_t *__restrict band, const bfloat16 *__restrict xa, float *__restrict c, int ky, int half) {
  mb_step_tile(band, ky, xa, c + half * 64 * MB_NT);
}
}
