#include "moe_combine.h"
extern "C" {
void mc_axpy(float *__restrict acc, const float *__restrict y0, const float *__restrict y1,
             const float *__restrict w, int e) {
  mc_axpy_impl(acc, y0, y1, w, e);
}
}
