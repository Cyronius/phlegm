// h band `band` = silu(g) * u, into one 1024 B output element
#include "xmb.h"
extern "C" {
void xmb_silu(const float *__restrict u, const float *__restrict g, float *__restrict h, int band) {
  xmb_silu_band(u + band * 64 * XMB_NT, g + band * 64 * XMB_NT, reinterpret_cast<bfloat16 *>(h));
}
}
