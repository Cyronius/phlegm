// SPDX-License-Identifier: MIT
#pragma once
#include <stdint.h>
#include <string.h>
namespace fsp {
static inline uint32_t bits(float x) { uint32_t u; memcpy(&u,&x,4); return u; }
static inline float value(uint32_t u) { float x; memcpy(&x,&u,4); return x; }
static inline float bf16(float x) {
  uint32_t u=bits(x); return value((u+0x7fffu+((u>>16)&1u))&0xffff0000u);
}
// Encode already BF16-rounded values. Integer RNE is independent of ambient rounding.
static inline int mantissa(float x,unsigned e) {
  uint32_t u=bits(x), a=u&0x7fffffffu;
  if(!a) return 0;
  unsigned ex=a>>23, sig=(a&0x7fffffu)| (ex ? 0x800000u : 0u);
  // BF16 values have only eight significant bits, including subnormals.
  unsigned shift=e+17-(ex ? ex : 1);
  unsigned q=0;
  if(shift<32) {
    unsigned base=sig>>shift, rem=sig&((1u<<shift)-1u), half=1u<<(shift-1);
    q=base+(rem>half || (rem==half && (base&1u)));
  }
  return u>>31 ? -int(q) : int(q);
}
static inline void encode8(const float *v,uint8_t *o,float *decoded) {
  uint32_t maxbits=0;
  for(unsigned i=0;i<8;i++) { uint32_t a=bits(v[i])&0x7fffffffu; if(a>maxbits)maxbits=a; }
  if(!maxbits) { for(unsigned i=0;i<9;i++)o[i]=0; for(unsigned i=0;i<8;i++)decoded[i]=0;return; }
  unsigned e=maxbits>>23; if(e<7)e=7;
  bool raise=false; for(unsigned i=0;i<8;i++)if(mantissa(v[i],e)>127)raise=true;
  if(raise)++e;
  o[0]=uint8_t(e);
  const float scale=value((e-6)<<23);
  for(unsigned i=0;i<8;i++) { int q=mantissa(v[i],e);o[i+1]=uint8_t(q);decoded[i]=float(q)*scale; }
}
static inline void pack8(const float *x,uint8_t *hi,uint8_t *lo) {
  float v[8],d[8],r[8],unused[8];
  for(unsigned i=0;i<8;i++)v[i]=bf16(x[i]);
  encode8(v,hi,d);
  for(unsigned i=0;i<8;i++)r[i]=bf16(v[i]-d[i]);
  encode8(r,lo,unused);
}
// FP32 polynomial exp(-abs(g)); bounded exponent, fifth-degree Taylor on [0,ln2].
// Prototype scalar arithmetic, deliberately not an optimized throughput claim.
static inline float silu_up(float g,float up) {
  float a=g<0 ? -g : g;
  float z=0;
  if(a<80.0f) {
    float x=-a*1.4426950408889634f;
    int n=int(x); if(float(n)>x)--n;
    float r=(x-float(n))*0.6931471805599453f;
    float p=1.0f+r*(1.0f+r*(0.5f+r*(0.1666666666666667f+r*(0.0416666666666667f+r*0.0083333333333333f))));
    z=p*value(uint32_t(n+127)<<23);
  }
  float s=1.0f/(1.0f+z);
  if(g<0)s=z*s;
  return (g*s)*up;
}
}
