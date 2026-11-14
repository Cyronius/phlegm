#include "fused2.h"
// The dispatch preamble, one element pair a call (st[1] counts them): pair 0 is the mask packet (one
// int32 segment ID for each of the dispatch's 256 tokens), expanded into this core's score bias;
// pairs 1-48 carry every core's input states, token-major [256][384] fp32, 8192 bytes a pair (stream
// A's 4096, then B's). This core keeps its 12288 bytes, six 2048-byte pieces, each reordered on arrival
// into x's [k/8][t][8] layout.
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
    // A piece starts at float 512 own of the core's [8][384]: token t0, chunk k0.
    static const int8 t0[6]={0,1,2,4,5,6},k0[6]={0,16,32,0,16,32};
    for(int p=0;p<4;++p) {
      const int own=4*(i-1)+p-6*core;
      if(own<0 || own>5) continue;
      const float *src=(const float *)(p<2 ? a+p*2048 : b+(p-2)*2048);
      int t=t0[own],k=k0[own];
#pragma clang loop unroll(disable)
      for(int blk=0;blk<64;++blk) {
        aie::store_v(x+k*64+t*8,aie::load_v<8>(src+blk*8));
        if(++k==48) { k=0;++t; }
      }
    }
  }
#ifdef F2_PROF_CYCLES
  if(F2_PROF_CYCLES & 16) f2_prof_last=F2_CYC();
#endif
}
