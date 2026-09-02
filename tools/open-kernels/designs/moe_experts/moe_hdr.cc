// moe_experts: the first element of every core's weight stream is the header:
// xm bf16[2048] at 0 and the router output f32[1024] at 4096 (idx int32[8] at
// float 256, w[8] at float 264). A core has only 2 input DMA channels (w and
// h), so xm/rout ride in the w stream; copied out because release() frees the
// oldest element of the fifo. rw gets floats 256..287 (w[e] = rw[8 + e]).
// One entry point per TU.
#include "vecmath.h"

extern "C" {
void moe_hdr(const uint8_t *__restrict e, bfloat16 *__restrict x, float *__restrict rw) {
  const bfloat16 *__restrict xs = (const bfloat16 *)e;
  const float *__restrict rs = (const float *)(e + 4096 + 1024);
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 2048; j += 32)
    aie::store_v(x + j, aie::load_v<32>(xs + j));
  aie::store_v(rw, aie::load_v<32>(rs));
}
}
