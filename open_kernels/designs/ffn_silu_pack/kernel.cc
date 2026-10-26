// SPDX-License-Identifier: MIT
#include "encoder.h"
extern "C" void ffn_silu_pack(const float *__restrict gate,const float *__restrict up,uint8_t *__restrict out) {
  // Inputs are [64 features][32 tokens]. Output one [K64][T32] 4608-byte block.
  for(unsigned nt=0;nt<4;nt++)
    for(unsigned ki=0;ki<8;ki++)
      for(unsigned row=0;row<8;row++) {
        float v[8];
        for(unsigned k=0;k<8;k++) {
          unsigned i=(ki*8+k)*32+nt*8+row;
          v[k]=fsp::silu_up(gate[i],up[i]);
        }
        unsigned p=(nt*8+ki)*144+row*9;
        fsp::pack8(v,out+p,out+p+72);
      }
}
