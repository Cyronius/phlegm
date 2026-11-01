#include "fused2.h"
extern "C" void f2_norm(float *x,const uint8 *p) {
#ifdef F2_STUB
  return;
#endif
 F2_T0(8)
 norm(x,(const float*)p);
 F2_T1(8, 9) }
