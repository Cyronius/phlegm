#include "fused2.h"
// The dispatch preamble, one element pair a call (st[1] counts them): pair 0 is the mask packet (one
// int32 segment ID per token), expanded into this core's score bias; pairs 1-24 carry every core's
// input states, token-major [128][384] fp32, 8192 bytes a pair (stream A's 4096, then B's), of which
// this core keeps its 6144: staged in w (4096 bytes) and q (2048), then reordered into x's
// [k/8][t][8] layout after the last pair.
extern "C" void f2_pre(const uint8 *a,const uint8 *b,float *x,float *w,bfloat16 *q,float *bias,int32 *st,int core) {
#ifdef F2_STUB
  return;
#endif
  const int i=st[1];
  st[1]=i==kPreamble-1 ? 0 : i+1;
  if(i==0) {
#ifdef F2_PROF_CYCLES
    for(int k=0;k<16;++k) f2_prof[k]=0;
    f2_prof[14]=F2_CYC();
#endif
    st[0]=0;
    attention_bias((const int32 *)a,bias,core);
  } else {
    // 2048-byte pieces: the pair holds pieces 4(i-1) .. 4(i-1)+3, this core owns 3 core .. 3 core+2.
    for(int p=0;p<4;++p) {
      const int own=4*(i-1)+p-3*core;
      if(own<0 || own>2) continue;
      const float *src=(const float *)(p<2 ? a+p*2048 : b+(p-2)*2048);
      float *dst=own<2 ? w+own*512 : (float *)q;
      for(int k=0;k<512;k+=32) aie::store_v(dst+k,aie::load_v<32>(src+k));
    }
    if(i==kPreamble-1)
      for(int t=0;t<4;++t)
        for(int k=0;k<48;++k) {
          const int g=t*384+k*8;      // the staged float index: w holds 0 .. 1023, q's bytes the rest
          aie::store_v(x+k*32+t*8,aie::load_v<8>(g<1024 ? w+g : (const float *)q+g-1024));
        }
  }
#ifdef F2_PROF_CYCLES
  if(F2_PROF_CYCLES & 16) f2_prof_last=F2_CYC();
#endif
}
