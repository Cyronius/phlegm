#include "fused2.h"
// A layer, one broadcast element pair a call; st[0] counts its kSteps steps:
//   0-17    K, V, Q (kind (i / 6 + 1) % 3), two packets (stream A's, stream B's) a step; quantizes x
//           first. K and V leave after step 13 (trit_design.py), so their exchange runs while the
//           last four Q steps compute
//   18-41   attention: head (i - 18) / 2, keys 64 ((i - 18) % 2) on: 32 from each stream (both carry
//           this core's slot)
//   42-47   O projection into the residual; quantizes the attention output first
//   48      first LayerNorm, then quantizes x for the FFN
//   49-108  FFN, six groups of ten: four up steps (256 GELU channels), then six down steps
//   109     second LayerNorm
static inline void step(const uint8 *a,const uint8 *b,float *x,bfloat16 *q,float *w,float *ml,
                        const bfloat16 *bias,int8 *aq,int8 *aq2,uint8 *kv,int i,int slot);
extern "C" void f2_step(const uint8 *a,const uint8 *b,float *x,bfloat16 *q,float *w,float *ml,
                        const bfloat16 *bias,int8 *aq,int8 *aq2,uint8 *kv,int32 *st,int slot) {
#ifdef F2_STUB
  return;
#endif
  const int i=st[0];
  st[0]=i==kSteps-1 ? 0 : i+1;
#ifdef F2_PROF_CYCLES
  if(F2_PROF_CYCLES & 16) f2_prof[i<18 ? 10 : i==18 ? 11 : i<42 ? 12 : (i>=49 && i<109) ? 13 : 0]+=F2_CYC()-f2_prof_last;
#endif
#ifdef F2_PROF_CYCLES
  const uint32 f2_ts=(F2_PROF_CYCLES & 64) ? F2_CYC() : 0;
#endif
  step(a,b,x,q,w,ml,bias,aq,aq2,kv,i,slot);
#ifdef F2_PROF_CYCLES
  if(F2_PROF_CYCLES & 64) f2_prof[11]+=F2_CYC()-f2_ts;
  if(F2_PROF_CYCLES & 16) f2_prof_last=F2_CYC();
#endif
}
// The plan, one entry a step: op, the projection kind or attention element (a's; b's is the next),
// the packet (pairs j, j + 1) or head, and when to quantize: x before (Q, K, V) or after (the first LayerNorm), the attention
// output before (O), the GELU output after (each group's fourth up step). A table instead of code:
// no divisions, one call site each.
enum : uint16 { kProject=0,kAttend=1,kNorm=2, kQxBefore=1<<9,kQqBefore=1<<10,kQxAfter=1<<11,kQq2After=1<<12,kD256=1<<13 };
static constexpr uint16 plan_entry(int i) {
  if(i<18) return kProject|((i/6+1)%3)<<2|(2*(i%6))<<5|(i==0 ? kQxBefore : 0);
  if(i<42) return kAttend|(2*((i-18)%2))<<2|((i-18)/2)<<5;
  if(i<48) return kProject|3<<2|(2*(i-42))<<5|(i==42 ? kQqBefore : 0);
  if(i==48) return kNorm|kQxAfter;
  if(i<109) {
    const int r=(i-49)%10;
    return r<4 ? kProject|4<<2|(2*r)<<5|(r==3 ? kQq2After : 0) : kProject|3<<2|(2*(r-4))<<5|kD256;
  }
  return kNorm;
}
struct Plan {
  uint16 v[kSteps];
  constexpr Plan() : v() { for(int i=0;i<kSteps;++i) v[i]=plan_entry(i); }
};
static constexpr Plan kPlan;

static inline void step(const uint8 *a,const uint8 *b,float *x,bfloat16 *q,float *w,float *ml,
                        const bfloat16 *bias,int8 *aq,int8 *aq2,uint8 *kv,int i,int slot) {
  const unsigned e=kPlan.v[i];
  const int op=e&3,kind=(e>>2)&7,j=(e>>5)&15;
  if(e&(kQxBefore|kQqBefore)) {
    F2_T0(2) quant(e&kQxBefore ? (const void *)x : (const void *)q,aq,384,!(e&kQxBefore)); F2_T1(2,e&kQxBefore ? 1 : 2)
  }
  if(op==kProject) {
    int8 *src=e&kD256 ? aq2 : aq;
    const int d=e&kD256 ? 256 : 384;
    F2_T0(1) project(a,src,x,q,kv,(float*)aq2,j,kind,d); project(b,src,x,q,kv,(float*)aq2,j+1,kind,d); F2_T1(1,3+kind)
  } else if(op==kAttend) {
    F2_T0(4) attend(q,a,w,ml,bias,j,kind); attend(q,b,w,ml,bias,j,kind+1); F2_T1(4,8)
  } else {
    F2_T0(8) norm(x,(const float *)a); F2_T1(8,9)
  }
  if(e&(kQxAfter|kQq2After)) {
    F2_T0(2) quant(e&kQxAfter ? (const void *)x : (const void *)q,e&kQxAfter ? aq : aq2,e&kQxAfter ? 384 : 256,!(e&kQxAfter)); F2_T1(2,e&kQxAfter ? 1 : 2)
  }
}
