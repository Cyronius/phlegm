#include "fused2.h"
extern "C" void f2_attend(bfloat16 *q,const uint8 *e,float *w,float *ml,
                          const float *bias,int head,int block) {
#ifdef F2_STUB
  return;
#endif
  F2_T0(4)
  attend(q,e,w,ml,bias,head,block);
  F2_T1(4, 8)
}
