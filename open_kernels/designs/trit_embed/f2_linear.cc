#include "fused2.h"
extern "C" void f2_linear(const uint8 *w,const int8 *aq,float *x,
                          bfloat16 *q,uint8 *kv,int j,int kind,int d) {
#ifdef F2_STUB
  return;
#endif
  F2_T0(1)
  // An element holds two packets: weight blocks 2j and 2j + 1.
#pragma clang loop unroll(disable)
  for(int s=0;s<2;++s) project(w+s*4096,aq,x,q,kv,2*j+s,kind,d);
  F2_T1(1, 3+kind)
#ifdef F2_PROF_CYCLES
  if (F2_PROF_CYCLES & 1) f2_prof[11]++;
#endif
}
