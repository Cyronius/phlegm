// attn_fin without a gate (ATTN_GATE=0 designs): og = o / l for one element of kHPO heads.
#include "attn.h"
extern "C" {
void attn_fin_ng(const float *__restrict oacc, const float *__restrict ml, bfloat16 *__restrict og, int hp) {
  attn_fin_impl(oacc, ml, nullptr, nullptr, og, hp);
}
}
