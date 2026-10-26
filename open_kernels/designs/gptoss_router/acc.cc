// BF16 GPT-OSS router weights are row-major [2880,32]. One call covers 8 rows.
#include "vecmath.h"

extern "C" void gptoss_router_acc(const bfloat16 *__restrict W,
                                   const bfloat16 *__restrict x,
                                   float *__restrict acc, int32_t rb) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  accf32 a;
  if (rb == 0) a = aie::zeros<accfloat, 32>();
  else a.from_vector(aie::load_v<32>(acc));
  const bfloat16 *xr = x + (unsigned)rb * 8;
  for (unsigned r = 0; r < 8; ++r)
    a = aie::mac(a, aie::load_v<32>(W + r * 32), xr[r]);
  aie::store_v(acc, a.template to_vector<float>());
}
