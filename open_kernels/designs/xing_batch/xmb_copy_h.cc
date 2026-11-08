// h k-tile `t` (64 k x XMB_NT tokens bf16) from the B fifo into the core's L1 copy of h
#include "xmb.h"
extern "C" {
void xmb_copy_h(const bfloat16 *__restrict e, bfloat16 *__restrict h, int t) {
  bfloat16 *__restrict d = h + t * 64 * XMB_NT;
  for (unsigned i = 0; i < 64 * XMB_NT; i += 32) aie::store_v(d + i, aie::load_v<32>(e + i));
}
}
