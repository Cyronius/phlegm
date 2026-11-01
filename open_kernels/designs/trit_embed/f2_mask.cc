#include "fused2.h"
// The mask packet carries one int32 segment ID per token; expand it to this core's score bias.
extern "C" void f2_mask(const uint8 *e,float *bias,int core) {
#ifdef F2_STUB
  return;
#endif
#ifdef F2_PROF_CYCLES
  for(int i=0;i<16;++i) f2_prof[i]=0;
  f2_prof[14]=F2_CYC();
#endif
  attention_bias((const int32 *)e,bias,core);
}
