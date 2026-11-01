#pragma once
// Native 384 post-norm BERT, four token rows per core. Adapted from the
// archived tiny-planner's tt.h / vecmath.h; no cross attention or pre-norm.
// fused2: attention masks come from per-token segment IDs (0 = padding), so any
// set of texts can share one 128-token dispatch. The mask is expanded once per
// dispatch into an additive score bias instead of being tested lane by lane.
// The broadcast moves 8192-byte elements: two 4096-byte weight packets (each with its 32
// output channels' bias and scale at byte 3072, packet format 2 below), 64 keys of one
// head's K/V, or one LayerNorm's parameters.
#define AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 1
#include "vecmath.h"
using MM = aie::mmul<4,8,8,bfloat16,bfloat16,accauto>;
#ifdef F2_PROF_CYCLES
// Profiling builds only: per-core cycle totals by kernel. f2_mask resets them each dispatch and
// f2_out writes them into the unused tail of the layer-trace packet (uint32 [16] at byte 6144).
// Slots: 1 quant_x, 2 quant_q, 3+kind linear (Q, K, V, residual, GELU), 8 attend, 9 norm,
// 11 linear calls, 14 dispatch start, 15 cycles from the mask to the last out.
__attribute__((weak)) uint32 f2_prof[16];
// F2_PROF_CYCLES is a bit mask of the wrappers that time themselves (program memory has room for
// only some at once): 1 linear, 2 quantization, 4 attend, 8 norm.
// get_cycles() has no definition for aie2p: read the memory module's Timer_Low (0x140F8) over the
// processor bus, which the runtime sequence enables in profiling builds (trit_design.py), between
// scheduling barriers (layer_x/gen_kernels.py has the same reader).
static inline __attribute__((always_inline)) uint32 f2_now() {
  __builtin_aie2p_sched_barrier();
  const uint32 v = (uint32)__builtin_aie2p_read_tm((int *)(0x80000 + 0x140F8));
  __builtin_aie2p_sched_barrier();
  return v;
}
#define F2_CYC() f2_now()
#define F2_T0(bit) const uint32 f2_t0 = (F2_PROF_CYCLES & (bit)) ? F2_CYC() : 0;
#define F2_T1(bit, slot) if (F2_PROF_CYCLES & (bit)) f2_prof[slot] += F2_CYC() - f2_t0;
#else
#define F2_T0(bit)
#define F2_T1(bit, slot)
#endif
using TM = aie::mmul<4,16,16,int8,int4>;
static inline v32b bf(v32f v) { accf32 a; a.from_vector(v); return a.to_vector<bfloat16>(); }
static inline v32f fp(v32b v) { accf32 a; a.from_vector(v); return a.to_vector<float>(); }
static inline v32f rep(const float *p) { auto v=aie::load_v<8>(p); return aie::concat(v,v,v,v); }
static inline v32f rows(const float *p) {
  return aie::concat(aie::broadcast<float,8>(p[0]),aie::broadcast<float,8>(p[1]),
                     aie::broadcast<float,8>(p[2]),aie::broadcast<float,8>(p[3]));
}
static inline float sum8(const float *p) { return aie::reduce_add(aie::concat(aie::load_v<8>(p),aie::zeros<float,8>())); }

static inline v32f src(const void *a,int kb,int isbf) {
  return isbf ? fp(aie::load_v<32>((const bfloat16*)a+kb*32)) : aie::load_v<32>((const float*)a+kb*32);
}

inline __attribute__((noinline)) void quant(const void *a,int8 *q,int d,int isbf) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  v32f mx=aie::zeros<float,32>();
  // |x| by clearing the sign bit: aie::abs on fp32 lanes compiles to 32 scalar extracts and inserts.
  const auto nosign=aie::broadcast<int32,32>(0x7fffffff);
  AIE_LOOP_MIN_ITERATION_COUNT(32)
  for(int kb=0;kb<d/8;++kb) mx=aie::max(mx,aie::bit_and(src(a,kb,isbf).cast_to<int32>(),nosign).cast_to<float>());
  alignas(64) float tmp[32], scales[32];
  aie::store_v(tmp,mx);
  aie::store_v(scales,aie::broadcast<float,32>(1.f));
  for(int r=0;r<4;++r) scales[r]=aie::reduce_max(aie::concat(aie::load_v<8>(tmp+r*8),aie::zeros<float,8>()));
  auto bits=aie::max(aie::load_v<32>(scales),aie::broadcast<float,32>(1e-5f)).cast_to<int32>();
  auto e=aie::add(aie::sub(aie::downshift(bits,23),aie::broadcast<int32,32>(133)),
    aie::select(aie::zeros<int32,32>(),aie::broadcast<int32,32>(1),
      aie::gt(aie::bit_and(bits,aie::broadcast<int32,32>(0x7fffff)),aie::broadcast<int32,32>(0x7e0000))));
  aie::store_v((float*)(q+4*d),aie::upshift(aie::add(e,aie::broadcast<int32,32>(127)),23).cast_to<float>());
  alignas(64) int32 ee[32];
  aie::store_v(ee,aie::upshift(aie::sub(aie::zeros<int32,32>(),e),23));
  // Add exponent bits to multiply fp32 by 2^-e exactly, retaining its mantissa.
  const auto iv=aie::concat(aie::broadcast<int32,8>(ee[0]),aie::broadcast<int32,8>(ee[1]),
                            aie::broadcast<int32,8>(ee[2]),aie::broadcast<int32,8>(ee[3]));
  aie::accum<acc32,32> s0=aie::zeros<acc32,32>(),s1=aie::zeros<acc32,32>();
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for(int kb=0;kb<d/16;++kb) {
    auto x0=aie::add(src(a,2*kb,isbf).cast_to<int32>(),iv).cast_to<float>();
    auto x1=aie::add(src(a,2*kb+1,isbf).cast_to<int32>(),iv).cast_to<float>();
    auto [z0,z1]=aie::interleave_zip(x0,x1,8);
    auto c0=aie::to_fixed<int16>(z0,0),c1=aie::to_fixed<int16>(z1,0);
    auto k0=aie::min(aie::max(c0,(int16)-128),(int16)127),k1=aie::min(aie::max(c1,(int16)-128),(int16)127);
    aie::store_v(q+kb*64,aie::pack(k0));
    aie::store_v(q+kb*64+32,aie::pack(k1));
    s0=aie::add(s0,k0);s1=aie::add(s1,k1);
  }
  // Each token's sum of its int8 codes, after the scales: the projections' offset-coded
  // fields add it once per output channel (project).
  const auto v0=s0.to_vector<int32>(),v1=s1.to_vector<int32>();
  int32 *sums=(int32*)(q+4*d+64);
  sums[0]=aie::reduce_add(v0.extract<16>(0));sums[1]=aie::reduce_add(v0.extract<16>(1));
  sums[2]=aie::reduce_add(v1.extract<16>(0));sums[3]=aie::reduce_add(v1.extract<16>(1));
}

// 1/d for the attention normalization: hardware estimate refined by one Newton step.
inline __attribute__((noinline)) v32f frecip(v32f d) {
  v32f r=aie::inv(d);
  return fmulN<32>(r,fsubN<32>(aie::broadcast<float,32>(2.f),fmulN<32>(d,r)));
}

#include "gelu_lut.h"
// tanh GELU by table (the hardware LUT's linear approximation): the input rounds to bf16, then one
// of 256 linear pieces over [-8, 8) from gelu_lut.py, worst error 2.2e-4 against the exact GELU of
// that bf16 input. The output is bf16, as the next quantization reads it.
static inline v32b gelu_bf(v32f y) {
  using lut_type = aie::lut<4, float, bfloat16>;
  const lut_type lut(kGeluPieces, (bfloat16 *)gelu_lut_ab, (bfloat16 *)gelu_lut_cd);
  const int step_bits = -4;
  aie::linear_approx<bfloat16, lut_type> la(lut, step_bits, kGeluPieces / 2, 0);
  const v32b x = bf(y);
  return aie::concat(la.compute(x.extract<16>(0)).to_vector<bfloat16>(),
                     la.compute(x.extract<16>(1)).to_vector<bfloat16>());
}

// Packet format 2 (trit_npu.weight_stream): the byte at (K block kb, lane) holds four 2-bit fields,
// channels 2l (bits 0-1), 2l+16 (2-3), 2l+1 (4-5), 2l+17 (6-7), l = lane % 8. The fields of channels
// 0-15 store code + 1, so masking them (0x33) gives int4 values 0..2 with no shift; their products
// carry one extra sum of the token's codes, removed with the sums quant leaves after its scales.
// Channels 16-31 keep the code at bits 2-3 of the nibble (0xcc: 4 * code). At byte 3072 the
// four 8-channel chunks' bias (fp32) and then weight scale (bf16; scale for channels 0-15,
// scale / 4 for 16-31) are stored already repeated for the 4 token rows. y = bias + acc * scale *
// the token's power-of-two activation scale, from bf16 hi/lo halves of the exact integer acc.
static inline v32f epi_chunk(v32f u,const float *bias,const bfloat16 *ws,v32b sa) {
  v32b uh,ul;
  split32(u,uh,ul);
  const v32b scale=aie::mul(aie::load_v<32>(ws),sa).to_vector<bfloat16>();
  accf32 v;
  v.from_vector(aie::load_v<32>(bias));
  v=aie::mac(v,uh,scale);
  v=aie::mac(v,ul,scale);
  return v.to_vector<float>();
}

// kind: Q=0, K transpose=1, V=2, residual=3, GELU=4. Profiling switches (TRIT_DEFS): F2_PROF_NOMAC
// keeps one MAC step, F2_PROF_NOEPI skips the float epilogue, F2_PROF_NOGELU treats GELU as linear.
inline __attribute__((noinline)) void project(const uint8 *__restrict w,const int8 *__restrict aq,
    float *__restrict x,bfloat16 *__restrict q,uint8 *__restrict kv,int j,int kind,int d) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  TM lo,hi;
#ifdef F2_PROF_NOMAC
  for(int kb=0;kb<1;++kb) {
#else
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for(int kb=0;kb<d/16;++kb) {
#endif
    auto a=aie::load_v<64>(aq+kb*64);
    auto b=aie::load_v<128>(w+kb*128);
    lo.mac(a,aie::bit_and((uint8)0x33,b).cast_to<int4>());
    hi.mac(a,aie::bit_and((uint8)0xcc,b).cast_to<int4>());
  }
  const int32 *sum=(const int32*)(aq+4*d+64);
  const auto off=aie::concat(aie::broadcast<int32,16>(sum[0]),aie::broadcast<int32,16>(sum[1]),
                             aie::broadcast<int32,16>(sum[2]),aie::broadcast<int32,16>(sum[3]));
  const auto lv=aie::sub(lo.to_vector<int32>(),off),hv=hi.to_vector<int32>();
#ifdef F2_PROF_NOEPI
  aie::store_v((int32*)q+(j%6)*128,lv);aie::store_v((int32*)q+(j%6)*128+64,hv);
  return;
#endif
  auto [u0,u1]=aie::interleave_unzip(aie::to_float<float>(lv.extract<32>(0),0),aie::to_float<float>(lv.extract<32>(1),0),8);
  auto [u2,u3]=aie::interleave_unzip(aie::to_float<float>(hv.extract<32>(0),0),aie::to_float<float>(hv.extract<32>(1),0),8);
  const float *bias=(const float*)(w+3072);
  const bfloat16 *ws=(const bfloat16*)(w+3584);
  const v32b sa=bf(rows((const float*)(aq+4*d)));
  const v32f u[4]={u0,u1,u2,u3};
#pragma clang loop unroll(disable)
  for(int c=0;c<4;++c) {
    const v32f y=epi_chunk(u[c],bias+c*32,ws+c*32,sa);
    const int off=j*128+c*32;
    if(kind==3) aie::store_v(x+off,faddN<32>(aie::load_v<32>(x+off),y));
    else if(kind==1) aie::store_v((bfloat16*)kv+j*256+c*32,aie::transpose(bf(y),4,8));
    else if(kind==2) aie::store_v((bfloat16*)kv+j*256+128+c*32,bf(y));
#ifdef F2_PROF_NOGELU
    else aie::store_v(q+off,bf(y));
#else
    else aie::store_v(q+off,kind==4 ? gelu_bf(y) : bf(y));
#endif
  }
}

inline __attribute__((noinline)) v16f rsqrt(v16f x) {
  auto r=aie::sub(aie::broadcast<int32,16>(0x5f3759df),aie::downshift(x.cast_to<int32>(),1)).cast_to<float>();
  auto half=fscaleN<16>(x,.5f);
  for(int i=0;i<4;++i) r=fmulN<16>(r,fsubN<16>(aie::broadcast<float,16>(1.5f),fmulN<16>(half,fmulN<16>(r,r))));
  return r;
}

inline __attribute__((noinline)) void norm_sum(const float *x,float *tmp) {
  accf32 s=aie::zeros<accfloat,32>();
  for(int k=0;k<48;++k) s=aie::add(s,aie::load_v<32>(x+k*32));
  aie::store_v(tmp,s.to_vector<float>());
}
inline __attribute__((noinline)) void norm_rows(const float *tmp,float *dest,float eps) {
  for(int r=0;r<16;++r) dest[r]=r<4?sum8(tmp+r*8):384.f;
  aie::store_v(dest,faddN<16>(fscaleN<16>(aie::load_v<16>(dest),1.f/384.f),aie::broadcast<float,16>(eps)));
}
inline __attribute__((noinline)) void norm_sq(const float *x,const float *mean,float *tmp) {
  auto m=rows(mean);accf32 s=aie::zeros<accfloat,32>();
  for(int k=0;k<48;++k) { auto d=fsubN<32>(aie::load_v<32>(x+k*32),m);s=aie::add(s,fmulN<32>(d,d)); }
  aie::store_v(tmp,s.to_vector<float>());
}
inline __attribute__((noinline)) void norm_out(float *x,const float *p,const float *mean,const float *var) {
  auto m=rows(mean),rr=rows(var);
  for(int k=0;k<48;++k) {
    auto y=faddN<32>(fmulN<32>(fmulN<32>(fsubN<32>(aie::load_v<32>(x+k*32),m),rr),rep(p+k*8)),rep(p+512+k*8));
    aie::store_v(x+k*32,y);
  }
}
inline __attribute__((noinline)) void norm(float *x,const float *p) {
  alignas(64) float tmp[32],mean[16],var[16];
  norm_sum(x,tmp);norm_rows(tmp,mean,0.f);norm_sq(x,mean,tmp);norm_rows(tmp,var,1e-12f);
  aie::store_v(var,rsqrt(aie::load_v<16>(var)));
  norm_out(x,p,mean,var);
}

alignas(64) static const bfloat16 ones[64]={
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

inline __attribute__((noinline)) void finish_attention(float *work,bfloat16 *q,int head) {
  auto inv=frecip(aie::max(aie::load_v<32>(work+128),1e-20f));
  for(int d=0;d<4;++d) aie::store_v(q+head*128+d*32,bf(fmulN<32>(aie::load_v<32>(work+d*32),inv)));
}

// bias[i], i = (key/8)*32 + (row%4)*8 + key%8: 0 where query row core*4+row%4 may attend to
// key (same nonzero segment), else -1e30. attend reads it 8 keys (a group) at a time.
inline __attribute__((noinline)) void attention_bias(const int32 *seg,float *bias,int core) {
  for(int i=0;i<512;++i) {
    int key=(i>>7)*32+((i>>5)&3)*8+(i&7),row=core*4+((i>>3)&3);
    bias[i]=seg[key] && seg[key]==seg[row] ? 0.f : -1e30f;
  }
}

// Attention scores in the log2 domain, so softmax needs 2^x: the 1/sqrt(32) score scale and log2(e)
// are one constant, and fexp2 is 2^n * p(f), f = x - n in [-0.5, 0.5], with a degree-3 polynomial
// (1.4e-4 relative, against the 2e-3 of the bf16 the probabilities are rounded to). Each Horner step
// is one accumulator: c + p*f from bf16 hi/lo halves, with f split once.
static constexpr float kScoreLog2 = .1767766952966369f * 1.4426950408889634f;
static inline v32f horner(v32f p, v32b fh, v32b fl, float c) {
  v32b ph, pl;
  split32(p, ph, pl);
  accf32 acc;
  acc.from_vector(aie::broadcast<float,32>(c));
  acc = aie::mac(acc, ph, fh);
  acc = aie::mac(acc, ph, fl);
  acc = aie::mac(acc, pl, fh);
  return acc.to_vector<float>();
}
static inline __attribute__((always_inline)) v32f fexp2(v32f x) {
  x = aie::max(x, -126.f);
  const auto n = aie::to_fixed<int32>(x, 0);
  const v32f f = fsubN<32>(x, aie::to_float<float>(n, 0));
  v32b fh, fl;
  split32(f, fh, fl);
  v32f p = horner(aie::broadcast<float,32>(0.0550293090f), fh, fl, 0.2422568020f);
  p = horner(p, fh, fl, 0.6932530361f);
  p = horner(p, fh, fl, 0.9999513472f);
  return aie::add(p.cast_to<int32>(), aie::upshift(n, 23)).cast_to<float>();
}

// One head's online softmax over one element: kGroups groups of 8 keys (2 cores' 4 tokens) = 64
// keys. After the last element of the head (block 1) its output replaces its queries in q.
// Profiling switches: F2_PROF_NOEXP replaces the exponentials with their input, F2_PROF_NOMM skips
// the matrix products.
static constexpr int kGroups=8,kLastBlock=1;
inline __attribute__((noinline)) void attend(bfloat16 *q,const uint8 *packet,float *work,
    float *ml,const float *bias,int head,int block) {
  const auto *e=(const bfloat16*)packet;
  float *s=work+160;
  bfloat16 *p=(bfloat16*)(work+160+kGroups*32);
  if(block==0) {
    for(int i=0;i<160;i+=32) aie::store_v(work+i,aie::zeros<float,32>());
    ml[0]=-1e30f;
  }
  v32f maxv=aie::broadcast<float,32>(ml[0]);
  for(int group=0;group<kGroups;++group) {
    MM mm(aie::zeros<accfloat,32>());
    for(int d=0;d<4;++d) {
      auto [b0,b1]=aie::interleave_zip(aie::load_v<32>(e+group*512+d*32),aie::load_v<32>(e+group*512+256+d*32),4);
#ifndef F2_PROF_NOMM
      mm.mac(aie::load_v<32>(q+head*128+d*32),aie::concat(b0,b1));
#endif
    }
    auto sv=faddN<32>(fscaleN<32>(mm.to_vector<float>(),kScoreLog2),aie::load_v<32>(bias+block*kGroups*32+group*32));
    aie::store_v(s+group*32,sv);
    maxv=aie::max(maxv,sv);
  }
  float next=aie::reduce_max(maxv);
  auto mn=aie::broadcast<float,32>(next);
  // The groups' exponentials and the rescale factor as one loop over kGroups + 1 vectors, so that
  // Peano can overlap their dependency chains (the rest of work is free).
  float *ex=work+160+kGroups*48;
  for(int g=0;g<kGroups;++g) aie::store_v(ex+g*32,fsubN<32>(aie::load_v<32>(s+g*32),mn));
  aie::store_v(ex+kGroups*32,fsubN<32>(aie::broadcast<float,32>(ml[0]),mn));
  AIE_LOOP_MIN_ITERATION_COUNT(5)
#pragma clang loop unroll(disable)
#ifdef F2_PROF_NOEXP
  for(int i=0;i<kGroups+1;++i) {}
#else
  for(int i=0;i<kGroups+1;++i) aie::store_v(ex+i*32,fexp2(aie::load_v<32>(ex+i*32)));
#endif
  const v32f rb=aie::load_v<32>(ex+kGroups*32);
  v32b rbh,rbl;     // fmulN with rb split once
  split32(rb,rbh,rbl);
  for(int i=0;i<160;i+=32) {
    v32b wh,wl;
    split32(aie::load_v<32>(work+i),wh,wl);
    accf32 acc=aie::mul(wh,rbh);
    acc=aie::mac(acc,wh,rbl);
    acc=aie::mac(acc,wl,rbh);
    aie::store_v(work+i,acc.to_vector<float>());
  }
  ml[0]=next;
  for(int group=0;group<kGroups;++group) {
    auto sv=aie::load_v<32>(s+group*32);
    auto prob=aie::select(aie::zeros<float,32>(),aie::load_v<32>(ex+group*32),aie::gt(sv,-1e20f));
    aie::store_v(p+group*32,bf(prob));
  }
  for(int d=0;d<5;++d) {
    MM mm(accf32(aie::load_v<32>(work+d*32)));
    for(int group=0;group<kGroups;++group) {
      auto vv=d==4 ? aie::load_v<64>(ones) :
        aie::concat(aie::load_v<32>(e+group*512+128+d*32),aie::load_v<32>(e+group*512+384+d*32));
#ifndef F2_PROF_NOMM
      mm.mac(aie::load_v<32>(p+group*32),vv);
#endif
    }
    aie::store_v(work+d*32,mm.to_vector<float>());
  }
  if(block==kLastBlock) finish_attention(work,q,head);
}
