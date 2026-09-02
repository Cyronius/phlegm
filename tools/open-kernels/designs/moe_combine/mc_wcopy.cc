#include "moe_combine.h"
extern "C" {
void mc_wcopy(const float *__restrict rout, float *__restrict w) { mc_wcopy_impl(rout, w); }
}
