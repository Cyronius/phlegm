#include "fused2.h"
// The dispatch preamble, one element pair a call (st[1] counts them): pair 0 is the mask packet (one
// int32 segment ID for each of the dispatch's 256 tokens), expanded into this core's score bias;
// pairs 1-24 carry the input states of the slot's 16 cores (the streams are the slot's own),
// [core][k/8][t][8] fp32 (each core's 12288 bytes already in x's layout; trit_npu.submit), 8192 bytes a
// pair (stream A's 4096, then B's). This core copies its six 2048-byte pieces as they pass.
extern "C" void f2_pre(const uint8 *a,const uint8 *b,float *x,bfloat16 *bias,int32 *st,int core) {
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
    for(int p=0;p<4;++p) {
      const int own=4*(i-1)+p-6*(core&15);
      if(own<0 || own>5) continue;
      const float *src=(const float *)(p<2 ? a+p*2048 : b+(p-2)*2048);
      float *dst=x+own*512;
      AIE_LOOP_MIN_ITERATION_COUNT(8)
      for(int k=0;k<512;k+=64) {
        aie::store_v(dst+k,aie::load_v<16>(src+k));aie::store_v(dst+k+16,aie::load_v<16>(src+k+16));
        aie::store_v(dst+k+32,aie::load_v<16>(src+k+32));aie::store_v(dst+k+48,aie::load_v<16>(src+k+48));
      }
    }
  }
#ifdef F2_PROF_CYCLES
  if(F2_PROF_CYCLES & 16) f2_prof_last=F2_CYC();
  if((F2_PROF_CYCLES & 32) && i==kPreamble-1) f2_prof[0]=F2_CYC()-f2_prof[14];
#endif
}
