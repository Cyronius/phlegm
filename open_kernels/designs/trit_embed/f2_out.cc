#include "fused2.h"
// Layer trace in two packets: half h carries this core's tokens 4h .. 4h + 3, [4][384] fp32 in natural
// channel order (6144 bytes); profiling builds append their timer totals to half 1.
extern "C" void f2_out(const float *x,uint8 *e,int half) {
#ifdef F2_STUB
  return;
#endif
  float *o=(float *)e;
#pragma clang loop unroll(disable)
  for(int t=0;t<4;++t)
    for(int k=0;k<48;++k) aie::store_v(o+t*384+k*8,aie::load_v<8>(x+k*64+(4*half+t)*8));
#ifdef F2_PROF_CYCLES
  if(half) {
    f2_prof[15]=F2_CYC()-f2_prof[14];
    for(int i=0;i<16;++i) ((uint32*)(e+6144))[i]=f2_prof[i];
  }
#endif
}
