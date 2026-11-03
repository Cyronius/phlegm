// down: k-tile ky of a band into the core's y accumulator
#include "mb4.h"
extern "C" {
void mb_step_dn(const uint8_t *__restrict band, const bfloat16 *__restrict hb, float *__restrict c, int tile, int ky) {
  mb_step_tile(band, ky, hb + tile * 64 * MB_NT, c);
}
}
