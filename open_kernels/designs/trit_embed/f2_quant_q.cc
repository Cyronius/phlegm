#include "fused2.h"
extern "C" void f2_quant_q(const bfloat16 *x,int8 *q,int d) {
#ifdef F2_STUB
  return;
#endif
 F2_T0(2)
 quant(x,q,d,1);
 F2_T1(2, 2) }
