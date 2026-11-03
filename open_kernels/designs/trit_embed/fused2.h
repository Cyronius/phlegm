#pragma once
// Native 384 post-norm BERT, eight token rows per core: a dispatch carries two 128-token slots, the
// first on cores 0-15 and the second on cores 16-31 (core c holds tokens 8c .. 8c + 7). Attention masks
// come from per-token segment IDs (0 = padding), so any set of texts can share a slot. The broadcast
// runs on two streams of 4096-byte elements (trit_design.py): weight packets (packet format 3 below)
// in pairs, a LayerNorm's parameters, or for attention one slot's K and V on each stream.
//
// Activations are [k/8][t][8]: 8-channel chunks of the 8 tokens (64 values a chunk). The 4 x 8 MAC
// operands are a chunk's rows 0-3 or 4-7.
#include "vecmath.h"
#ifdef F2_PROF_CYCLES
// Profiling builds only: per-core cycle totals by kernel. f2_pre resets them each dispatch and f2_out
// writes them after the second half of the layer trace (uint32 [16] at byte 6144).
// Slots: 1 quant_x, 2 quant_q, 3+kind linear (Q, K, V, residual, GELU), 8 attend, 9 norm,
// 14 dispatch start, 15 cycles from the mask to the last out.
__attribute__((weak)) uint32 f2_prof[16];
__attribute__((weak)) uint32 f2_prof_last;
// F2_PROF_CYCLES is a bit mask of the wrappers that time themselves (program memory has room for
// only some at once): 1 linear, 2 quantization, 4 attend, 8 norm, 16 the waits between steps (slots
// 10 Q/K/V, 11 the first attention step (the K/V exchange), 12 the other attention steps, 13 FFN,
// 0 the rest).
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
// Broadcast schedule (trit_design.py): kPreamble element pairs a dispatch (f2_pre), kSteps a layer
// (f2_step). A head's K and V take kHeadBytes of a core's output packet (project, attend).
static constexpr int kPreamble=25,kSteps=110,kHeadBytes=576;
using acc64i = aie::accum<acc32,64>;
using acc64f = aie::accum<accfloat,64>;
static inline v32b bf(v32f v) { accf32 a; a.from_vector(v); return a.to_vector<bfloat16>(); }
static inline v64bfp16ebs8 bfp64(const acc64f &a) { return ::to_v64bfp16ebs8((v64accfloat)a); }
static inline acc64f ups64(aie::vector<bfloat16,64> v) { acc64f a; a.from_vector(v); return a; }
// 8 per-token values -> [t][8]; 8 per-channel values -> [t][8] (the same 8 for every token).
static inline aie::vector<float,64> rows8(const float *p) {
  return aie::concat(aie::concat(aie::broadcast<float,8>(p[0]),aie::broadcast<float,8>(p[1]),
                                 aie::broadcast<float,8>(p[2]),aie::broadcast<float,8>(p[3])),
                     aie::concat(aie::broadcast<float,8>(p[4]),aie::broadcast<float,8>(p[5]),
                                 aie::broadcast<float,8>(p[6]),aie::broadcast<float,8>(p[7])));
}
static inline aie::vector<float,64> rep8(const float *p) {
  const auto v=aie::load_v<8>(p);
  return aie::concat(aie::concat(v,v,v,v),aie::concat(v,v,v,v));
}
// bfp16 operands kept in memory as their two parts: 64 mantissa bytes, 8 exponent bytes.
static inline void store_bfp(int8 *m,uint32 *e,const v64bfp16ebs8 &b) {
  aie::store_v(m,aie::vector<int8,64>(b.mantissa));
  const v2int32 x=(v2int32)b.exponent;
  e[0]=x[0];e[1]=x[1];
}
static inline v64bfp16ebs8 load_bfp(const int8 *m,const uint32 *e) {
  v64bfp16ebs8 b;
  b.mantissa=aie::load_v<64>(m);
  const v2int32 x={(int)e[0],(int)e[1]};
  b.exponent=(v8int8)x;
  return b;
}

// Activation codes for the projections: int8 in the activations' own layout (each chunk's 32-byte
// halves are the 4 x 8 A operands of mac_4x8_8x16), then at q + 8d project's epilogue patterns, int32
// [t][16] for each row group: P_lo, P_hi (rows 0-3), P_lo, P_hi (rows 4-7). P_hi is the bits of
// 1.5 * 2^(23 + e) for the token's activation scale 2^e, P_lo is P_hi minus the token's sum of codes
// (see project). AQ holds 8d + 1024 bytes.
//
// The scale: m = max |x| over the token (exact: integer max of the sign-cleared bits), floored at
// 1e-5, gives e = exponent(m) - 6, plus 1 when m's mantissa would round past 127. The codes are
// round-half-even(x / 2^e) saturated to int8: x + 1.5 * 2^(23+e) in fp32 rounds at 2^e, and its
// bits minus the constant's are the integer.
template <typename T>
static inline acc64i quant_codes(const T *__restrict a,int8 *__restrict q,int d,acc64f mf,acc64i mi) {
  // Two chunks an iteration, with their own sums: the chain per chunk is long. The codes never
  // saturate (|x| / 2^e <= 127 by the choice of e), so the sums take them unrounded from the
  // accumulators.
  acc64i s0=aie::zeros<acc32,64>(),s1=aie::zeros<acc32,64>();
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for(int kb=0;kb<d/8;kb+=2) {
    acc64f x0,x1;
    x0.from_vector(aie::load_v<64>(a+kb*64));
    x1.from_vector(aie::load_v<64>(a+kb*64+64));
    const acc64i c0(::sub((v64acc32)aie::add(x0,mf).template cast_to<acc32>(),(v64acc32)mi));
    const acc64i c1(::sub((v64acc32)aie::add(x1,mf).template cast_to<acc32>(),(v64acc32)mi));
    aie::store_v(q+kb*64,c0.template to_vector<int8>());
    aie::store_v(q+kb*64+64,c1.template to_vector<int8>());
    s0=acc64i(::add((v64acc32)s0,(v64acc32)c0));
    s1=acc64i(::add((v64acc32)s1,(v64acc32)c1));
  }
  return acc64i(::add((v64acc32)s0,(v64acc32)s1));
}
inline __attribute__((noinline)) void quant(const void *a,int8 *q,int d,int isbf) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);
  alignas(64) int32 tmp[64];
  if(isbf) {
    aie::vector<int16,64> mx=aie::zeros<int16,64>();
    AIE_LOOP_MIN_ITERATION_COUNT(32)
    for(int kb=0;kb<d/8;++kb) mx=aie::max(mx,aie::bit_and((int16)0x7fff,aie::load_v<64>((const int16*)a+kb*64)));
    aie::store_v(tmp,aie::upshift(mx.unpack(),16));
  } else {
    aie::vector<int32,64> mx=aie::zeros<int32,64>();
    AIE_LOOP_MIN_ITERATION_COUNT(32)
    for(int kb=0;kb<d/8;++kb) mx=aie::max(mx,aie::bit_and((int32)0x7fffffff,aie::load_v<64>((const int32*)a+kb*64)));
    aie::store_v(tmp,mx);
  }
  int32 *pat=(int32*)(q+8*d);
  alignas(64) int32 mg[64];
  for(int t=0;t<8;++t) {
    int32 bits=aie::reduce_max(aie::load_v<8>(tmp+t*8));
    bits=bits>0x3727c5ac?bits:0x3727c5ac;                     // 1e-5f
    const int32 e=(bits>>23)-133+((bits&0x7fffff)>0x7e0000);
    const int32 m=(e+150)*(1<<23)+0x400000;                    // 1.5 * 2^(23+e)
    aie::store_v(mg+t*8,aie::broadcast<int32,8>(m));
    aie::store_v(pat+(t>>2)*128+64+(t&3)*16,aie::broadcast<int32,16>(m));
  }
  const aie::vector<int32,64> mv=aie::load_v<64>(mg);
  acc64f mf;
  mf.from_vector(mv.cast_to<float>());
  const acc64i mi(mv);
  const acc64i s=isbf ? quant_codes((const bfloat16*)a,q,d,mf,mi) : quant_codes((const float*)a,q,d,mf,mi);
  aie::store_v(tmp,s.to_vector<int32>());
  for(int t=0;t<8;++t) {
    int32 *p=pat+(t>>2)*128+(t&3)*16;
    aie::store_v(p,aie::broadcast<int32,16>(p[64]-aie::reduce_add(aie::load_v<8>(tmp+t*8))));
  }
}

#include "gelu_lut.h"
// tanh GELU by table: the input rounds to bf16, then one of 128 linear pieces over [-8, 8) from
// gelu_lut.py (4 KB of tables), against the exact GELU of that bf16 input worst error as gelu_lut.py
// prints. The output is bf16, as the next quantization reads it. This is aie::linear_approx<bfloat16,
// aie::lut<4, float, bfloat16>> (step bits -3, index bias 64) without its scratch-pad delay line, which
// Peano spills around every call: the same lookup, slope and offset shuffles and MAC.
static inline v16bfloat16 gelu16(v16bfloat16 x) {
  const int8 *ab=(const int8*)gelu_lut_ab+(kGeluPieces/2)*16,*cd=(const int8*)gelu_lut_cd+(kGeluPieces/2)*16;
  v16int32 idx=::bfloat16_to_int(x,kGeluShift);
  idx=::min(::max(idx,::broadcast_to_v16int32(-(int)(kGeluPieces/2)*16)),::broadcast_to_v16int32((int)(kGeluPieces/2)*16-1));
  v32bfloat16 c0,c1;
  ::load_lut_2x_float(ab,cd,idx,c0,c1);
  v32accfloat acc=::set_v32accfloat(1,(v16accfloat)::shuffle(c0,c1,T32_16x2_hi));
  acc=::mac_elem_32(::shuffle(c0,c1,T16_16x4_lo),::set_v32bfloat16(1,x),acc);
  return ::to_v16bfloat16(::extract_v16accfloat(acc,1));
}
static inline aie::vector<bfloat16,64> gelu64(acc64f y) {
  const aie::vector<bfloat16,64> x=y.to_vector<bfloat16>();
  return aie::concat(aie::vector<bfloat16,16>(gelu16(x.extract<16>(0))),aie::vector<bfloat16,16>(gelu16(x.extract<16>(1))),
                     aie::vector<bfloat16,16>(gelu16(x.extract<16>(2))),aie::vector<bfloat16,16>(gelu16(x.extract<16>(3))));
}

// Packet format 3 (trit_npu.weight_stream): the byte at (K block kb, lane) holds four 2-bit fields,
// channels 2l (bits 0-1), 2l+16 (2-3), 2l+1 (4-5), 2l+17 (6-7), l = lane % 8. The fields of channels
// 0-15 store code + 1, so masking them (0x33) gives int4 values 0..2 with no shift; their products
// carry one extra sum of the token's codes, which the P_lo pattern removes. Channels 16-31 keep the
// code at bits 2-3 of the nibble (0xcc: 4 * code). At byte 3072 come the bias (fp32) and then the
// weight scale (bf16; scale for channels 0-15, scale / 4 for 16-31), each laid out [half][t][16] like
// the two accumulators (half 0 = channels 0-15).
//
// The MAC runs on mac_4x8_8x16 directly, once for each row group: A is 4 tokens x 8 codes straight
// from quant's layout and B the unpacked int4 nibbles, one mask a K block, four MACs in four cycles.
// Adding quant's patterns (P_lo, P_hi) makes each int32 sum, read as fp32, 1.5 * 2^(23+e) + acc * 2^e
// exactly; subtracting 1.5 * 2^(23+e) leaves acc times the activation scale. y = bias + u * scale,
// with u split into bf16 hi/lo halves.
static inline acc64f epi_half(acc64i a,acc64f mf,const float *bias,const bfloat16 *ws) {
  const acc64f u=aie::sub(a.cast_to<accfloat>(),mf);
  const aie::vector<bfloat16,64> uh=u.to_vector<bfloat16>();
  const aie::vector<bfloat16,64> ul=aie::sub(u,uh).to_vector<bfloat16>();
  const aie::vector<bfloat16,64> w=aie::load_v<64>(ws);
  acc64f v;
  v.from_vector(aie::load_v<64>(bias));
  v=aie::mac(v,uh,w);
  return aie::mac(v,ul,w);
}

// A row group's [t][16]: its two 8-channel chunks [4][8], at p and p + 64 (one chunk on).
template <typename T>
static inline void store_unzip(T *p,aie::vector<T,64> v) {
  auto [c0,c1]=aie::interleave_unzip(v.template extract<32>(0),v.template extract<32>(1),8);
  aie::store_v(p,c0);aie::store_v(p+64,c1);
}

// kind: Q=0, K=1, V=2, residual=3, GELU=4. Q and GELU outputs are bf16 in the activation layout. A
// head's K and V go out in its slot of the output packet as bfp16, mantissas [chunk][64 bytes] (K at
// 0, V at 256), exponents [chunk][8 bytes] (K at 512, V at 544): K in blocks along the head dimension,
// V transposed (blocks along the tokens); row group 0's half waits in fp32 in park meanwhile. Profiling switches (TRIT_DEFS): F2_PROF_NOMAC keeps
// one MAC step, F2_PROF_NOEPI skips the float epilogue, F2_PROF_NOGELU treats GELU as linear.
inline __attribute__((noinline)) void project(const uint8 *__restrict w,const int8 *__restrict aq,
    float *__restrict x,bfloat16 *__restrict q,uint8 *__restrict kv,float *__restrict park,int j,int kind,int d) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *bias=(const float*)(w+3072);
  const bfloat16 *ws=(const bfloat16*)(w+3584);
  uint8 *head=kv+j*kHeadBytes;
#pragma clang loop unroll(disable)
  for(int g=0;g<2;++g) {
    // c: the raw bytes as int4, whose nibbles hold (low code + 1) + 4 * high code; h: the high fields
    // alone (0xcc). c - h is the low fields' sum. Both start at zero: starting c from P_lo instead
    // costs the loop a fifth cycle a K block (4 MACs in 4 cycles as it stands).
    v64acc32 c=aie::zeros<acc32,64>(),h=aie::zeros<acc32,64>();
    const int8 *ag=aq+g*32;
#ifdef F2_PROF_NOMAC
    for(int kb=0;kb<1;++kb) {
#else
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for(int kb=0;kb<d/16;++kb) {
#endif
      const aie::vector<uint8,128> b=aie::load_v<128>(w+kb*128);
      const auto br=b.cast_to<int4>(),bh=aie::bit_and((uint8)0xcc,b).cast_to<int4>();
      v64int8 a0,a1;
      a0=::set_v64int8(0,aie::load_v<32>(ag+kb*128)); a1=::set_v64int8(0,aie::load_v<32>(ag+kb*128+64));
      c=::mac_4x8_8x16(a0,::unpack(br.extract<128>(0)),c);
      h=::mac_4x8_8x16(a0,::unpack(bh.extract<128>(0)),h);
      c=::mac_4x8_8x16(a1,::unpack(br.extract<128>(1)),c);
      h=::mac_4x8_8x16(a1,::unpack(bh.extract<128>(1)),h);
    }
    const int32 *pat=(const int32*)(aq+8*d)+g*128;
    const acc64i lo=::add(::sub(c,h),acc64i(aie::load_v<64>(pat))),hi=::add(h,acc64i(aie::load_v<64>(pat+64)));
#ifdef F2_PROF_NOEPI
    aie::store_v((int32*)q+(j%6)*256+g*128,lo.to_vector<int32>());
    aie::store_v((int32*)q+(j%6)*256+g*128+64,hi.to_vector<int32>());
    continue;
#endif
    acc64f mf;
    mf.from_vector(aie::load_v<64>(pat+64).cast_to<float>());
    const acc64f y0=epi_half(lo,mf,bias,ws);
    const acc64f y1=epi_half(hi,mf,bias+64,ws+64);
    const int off=j*256+g*32;        // chunk 4j, this row group
    switch(kind) {
    case 3: {
      const aie::vector<float,64> f0=y0.to_vector<float>(),f1=y1.to_vector<float>();
      auto [a0,a1]=aie::interleave_unzip(f0.extract<32>(0),f0.extract<32>(1),8);
      auto [a2,a3]=aie::interleave_unzip(f1.extract<32>(0),f1.extract<32>(1),8);
      aie::store_v(x+off,faddN<32>(aie::load_v<32>(x+off),a0));
      aie::store_v(x+off+64,faddN<32>(aie::load_v<32>(x+off+64),a1));
      aie::store_v(x+off+128,faddN<32>(aie::load_v<32>(x+off+128),a2));
      aie::store_v(x+off+192,faddN<32>(aie::load_v<32>(x+off+192),a3));
      break;
    }
    case 1:
    case 2: {
      // park: [chunk][rows 0-3 x 8] until row group 1
      const aie::vector<float,64> f0=y0.to_vector<float>(),f1=y1.to_vector<float>();
      auto [a0,a1]=aie::interleave_unzip(f0.extract<32>(0),f0.extract<32>(1),8);
      auto [a2,a3]=aie::interleave_unzip(f1.extract<32>(0),f1.extract<32>(1),8);
      if(g==0) {
        aie::store_v(park,a0);aie::store_v(park+32,a1);aie::store_v(park+64,a2);aie::store_v(park+96,a3);
      } else {
        // K: chunk [8 tokens][8 dims]; V: transposed, [8 dims][8 tokens].
        int8 *m=(int8*)head+(kind-1)*256;
        uint32 *ex=(uint32*)(head+512)+(kind-1)*8;
        auto put=[&](int ch,aie::vector<float,32> r47) {
          const aie::vector<float,32> r03=aie::load_v<32>(park+ch*32);
          acc64f k;
          if(kind==1) k.from_vector(aie::concat(r03,r47));
          else {
            auto [t0,t1]=aie::interleave_zip(aie::transpose(r03,4,8),aie::transpose(r47,4,8),4);
            k.from_vector(aie::concat(t0,t1));
          }
          store_bfp(m+ch*64,ex+ch*2,bfp64(k));
        };
        put(0,a0);put(1,a1);put(2,a2);put(3,a3);
      }
      break;
    }
#ifdef F2_PROF_NOGELU
    case 4:
#endif
    case 0:
      store_unzip(q+off,y0.to_vector<bfloat16>());
      store_unzip(q+off+128,y1.to_vector<bfloat16>());
      break;
#ifndef F2_PROF_NOGELU
    default:
      store_unzip(q+off,gelu64(y0));
      store_unzip(q+off+128,gelu64(y1));
#endif
    }
  }
}

inline __attribute__((noinline)) v16f rsqrt(v16f x) {
  auto r=aie::sub(aie::broadcast<int32,16>(0x5f3759df),aie::downshift(x.cast_to<int32>(),1)).cast_to<float>();
  auto half=fscaleN<16>(x,.5f);
  for(int i=0;i<4;++i) r=fmulN<16>(r,fsubN<16>(aie::broadcast<float,16>(1.5f),fmulN<16>(half,fmulN<16>(r,r))));
  return r;
}

// LayerNorm (eps 1e-12) of each token's 384 channels: 48 chunks [8 tokens][8].
inline __attribute__((noinline)) void norm_sum(const float *x,float *tmp) {
  acc64f s=aie::zeros<accfloat,64>();
  for(int k=0;k<48;++k) s=aie::add(s,aie::load_v<64>(x+k*64));
  aie::store_v(tmp,s.to_vector<float>());
}
// The 8 rows' sums of [8][8] partial sums (pairs of lanes added three times), / 384, + eps, into
// dest[0..7] (dest[8..15] get the same).
inline __attribute__((noinline)) void norm_rows(const float *tmp,float *dest,float eps) {
  const aie::vector<float,64> v=aie::load_v<64>(tmp);
  const aie::vector<float,32> a=faddN<32>(aie::filter_even(v,1),aie::filter_odd(v,1));
  const aie::vector<float,16> b=faddN<16>(aie::filter_even(a,1),aie::filter_odd(a,1));
  const aie::vector<float,8> c=faddN<16>(aie::concat(aie::filter_even(b,1),aie::filter_even(b,1)),
                                         aie::concat(aie::filter_odd(b,1),aie::filter_odd(b,1))).extract<8>(0);
  aie::store_v(dest,faddN<16>(fscaleN<16>(aie::concat(c,c),1.f/384.f),aie::broadcast<float,16>(eps)));
}
// Products of fp32 values by bf16 hi/lo halves: d = dh + dl, so d * d = dh dh + 2 dh dl (+ dl dl, below
// fp32's rounding). The LayerNorm packet carries gamma already split (trit_npu.parameter_packet).
inline __attribute__((noinline)) void norm_sq(const float *__restrict x,const float *__restrict mean,float *__restrict tmp) {
  const auto m=rows8(mean);
  acc64f s0=aie::zeros<accfloat,64>(),s1=aie::zeros<accfloat,64>();
  for(int k=0;k<48;k+=2) {
    acc64f d0,d1;
    d0.from_vector(fsubN<64>(aie::load_v<64>(x+k*64),m));
    d1.from_vector(fsubN<64>(aie::load_v<64>(x+k*64+64),m));
    const aie::vector<bfloat16,64> dh0=d0.to_vector<bfloat16>(),dl0=aie::sub(d0,dh0).to_vector<bfloat16>();
    const aie::vector<bfloat16,64> dh1=d1.to_vector<bfloat16>(),dl1=aie::sub(d1,dh1).to_vector<bfloat16>();
    s0=aie::mac(s0,dh0,dh0);s0=aie::mac(s0,dh0,dl0);s0=aie::mac(s0,dl0,dh0);
    s1=aie::mac(s1,dh1,dh1);s1=aie::mac(s1,dh1,dl1);s1=aie::mac(s1,dl1,dh1);
  }
  aie::store_v(tmp,aie::add(s0,s1).to_vector<float>());
}
static inline aie::vector<bfloat16,64> rep8b(const bfloat16 *p) {
  const auto v=aie::load_v<8>(p);
  return aie::concat(aie::concat(v,v,v,v),aie::concat(v,v,v,v));
}
// y = (x - mean) r gamma + beta, r = 1 / sqrt(var + eps): t = (x - mean) r, then t gamma, each product
// from hi/lo halves (gamma's from the packet, r's split once).
inline __attribute__((noinline)) void norm_out(float *__restrict x,const float *__restrict p,const float *__restrict mean,const float *__restrict var) {
  const auto m=rows8(mean);
  acc64f ra;
  ra.from_vector(rows8(var));
  const aie::vector<bfloat16,64> rh=ra.to_vector<bfloat16>(),rl=aie::sub(ra,rh).to_vector<bfloat16>();
  const bfloat16 *__restrict g=(const bfloat16*)p;
  for(int k=0;k<48;k+=2) {      // two chunks a pass, loaded before either is stored, so they overlap
    acc64f d0,d1;
    d0.from_vector(fsubN<64>(aie::load_v<64>(x+k*64),m));
    d1.from_vector(fsubN<64>(aie::load_v<64>(x+k*64+64),m));
    const aie::vector<bfloat16,64> dh0=d0.to_vector<bfloat16>(),dl0=aie::sub(d0,dh0).to_vector<bfloat16>();
    const aie::vector<bfloat16,64> dh1=d1.to_vector<bfloat16>(),dl1=aie::sub(d1,dh1).to_vector<bfloat16>();
    acc64f t0=aie::mul(dh0,rh),t1=aie::mul(dh1,rh);
    t0=aie::mac(t0,dh0,rl);t0=aie::mac(t0,dl0,rh);
    t1=aie::mac(t1,dh1,rl);t1=aie::mac(t1,dl1,rh);
    const aie::vector<bfloat16,64> th0=t0.to_vector<bfloat16>(),tl0=aie::sub(t0,th0).to_vector<bfloat16>();
    const aie::vector<bfloat16,64> th1=t1.to_vector<bfloat16>(),tl1=aie::sub(t1,th1).to_vector<bfloat16>();
    acc64f y0,y1;
    y0.from_vector(rep8(p+512+k*8));y1.from_vector(rep8(p+520+k*8));
    const aie::vector<bfloat16,64> gh0=rep8b(g+k*8),gl0=rep8b(g+384+k*8),gh1=rep8b(g+8+k*8),gl1=rep8b(g+392+k*8);
    y0=aie::mac(y0,th0,gh0);y0=aie::mac(y0,th0,gl0);y0=aie::mac(y0,tl0,gh0);
    y1=aie::mac(y1,th1,gh1);y1=aie::mac(y1,th1,gl1);y1=aie::mac(y1,tl1,gh1);
    aie::store_v(x+k*64,y0.to_vector<float>());
    aie::store_v(x+k*64+64,y1.to_vector<float>());
  }
}
inline __attribute__((noinline)) void norm(float *x,const float *p) {
  alignas(64) float tmp[64],mean[16],var[16];
  norm_sum(x,tmp);norm_rows(tmp,mean,0.f);norm_sq(x,mean,tmp);norm_rows(tmp,var,1e-12f);
  aie::store_v(var,rsqrt(aie::load_v<16>(var)));
  norm_out(x,p,mean,var);
}

// bias[i] (bf16), i = (key/8)*64 + row*8 + key%8 over the slot's 128 keys: 0 where the core's row may
// attend to the key (same nonzero segment), else -1000: finite, so that a core whose rows are all
// padding still takes ordinary exponentials, and far enough down that a masked key's exponential is
// 2^-126 (the saturated floor of exp2_64, nothing after the bfp16 rounding) against any key it is
// masked from. seg: the dispatch's 256 segment IDs.
inline __attribute__((noinline)) void attention_bias(const int32 *seg,bfloat16 *bias,int core) {
  const int32 *ss=seg+128*(core>>4),*rs=ss+8*(core&15);
  // Lanes [row][key] for rows 0-3 (r0) and 4-7 (r1): each row's segment ID repeated over the 8 keys.
  aie::vector<int32,32> r0,r1;
  for(int r=0;r<4;++r) {
    r0.insert(r,aie::broadcast<int32,8>(rs[r]));
    r1.insert(r,aie::broadcast<int32,8>(rs[4+r]));
  }
  const aie::vector<bfloat16,32> zero=aie::zeros<bfloat16,32>(),off=aie::broadcast<bfloat16,32>((bfloat16)-1000.f);
#pragma clang loop unroll(disable)
  for(int kb=0;kb<16;++kb) {
    const aie::vector<int32,8> k8=aie::load_v<8>(ss+kb*8);
    const aie::vector<int32,32> kv=aie::concat(k8,k8,k8,k8);
    const auto real=aie::neq(kv,(int32)0);
    aie::store_v(bias+kb*64,aie::select(off,zero,aie::eq(kv,r0) & real));
    aie::store_v(bias+kb*64+32,aie::select(off,zero,aie::eq(kv,r1) & real));
  }
}

// Attention scores in the log2 domain, so softmax needs 2^x: the 1/sqrt(32) score scale and log2(e)
// are one constant, folded into the queries.
static constexpr float kScoreLog2 = .1767766952966369f * 1.4426950408889634f;
// 2^(x - m) on 64 lanes for an integer m, with no accumulator constants (they would not
// stay in registers): constants enter as bf16 MAC operands. r = round(x) by the 1.5 * 2^23 magic
// constant, whose sum's low 16 bits are also r; f = x - r in [-0.5, 0.5]; 2^f by a cubic with
// p(0) = 1 (1e-4 relative) with f in bf16 (|error| < 7e-4 relative, against the bfp16 the
// probabilities are rounded to, 8-bit mantissas); and r - m added to the exponent bits, saturated at
// -126 (masked scores, attention_bias, end up there).
static inline acc64f exp2_64(const acc64f &x,int m) {
  using bv=aie::vector<bfloat16,64>;
  const bv one=aie::broadcast<bfloat16,64>(1.f);
  const acc64f t=aie::mac(x,one,aie::broadcast<bfloat16,64>(12582912.f));
  const acc64f r=aie::mac(t,one,aie::broadcast<bfloat16,64>(-12582912.f));
  const acc64f f=aie::sub(x,r);
  const aie::vector<int16,64> n=aie::filter_even(t.to_vector<float>().cast_to<int16>(),1);
  acc64i n23;
  n23.from_vector(aie::max(aie::sub(n,(int16)m),(int16)-126),23);
  const bv fh=f.to_vector<bfloat16>();
  const bfloat16 c1h=(bfloat16)0.69328289f,c1l=(bfloat16)(0.69328289f-(float)c1h);
  acc64f lo=aie::mul(fh,aie::broadcast<bfloat16,64>(c1h));
  lo=aie::mac(lo,fh,aie::broadcast<bfloat16,64>(c1l));
  lo=aie::mac(lo,one,one);
  acc64f hi=aie::mul(fh,aie::broadcast<bfloat16,64>((bfloat16)0.05500855f));
  hi=aie::mac(hi,one,aie::broadcast<bfloat16,64>((bfloat16)0.24221029f));
  const acc64f pf=aie::mac(lo,aie::mul(fh,fh).to_vector<bfloat16>(),hi.to_vector<bfloat16>());
  return acc64i(::add((v64acc32)pf.cast_to<acc32>(),(v64acc32)n23)).cast_to<accfloat>();
}

#ifndef ATT_TS
#define ATT_TS(i)
#endif
// out = o / row sums, into this head's chunks of q. The sums [8][8] repeat each row's sum across its
// 8 lanes, the same layout as o's chunks, so 1 / sums is the multiplier as it stands: a bit-trick seed
// (12%) and three Newton steps r = r (2 - s r); then o * r from bf16 hi/lo halves.
inline __attribute__((noinline)) void finish_attention(const float *work,bfloat16 *q,int head) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const aie::vector<float,64> sv=aie::max(aie::load_v<64>(work+256),aie::broadcast<float,64>(1e-20f));
  aie::vector<float,64> y=aie::sub(aie::broadcast<int32,64>(0x7ef311c3),sv.cast_to<int32>()).cast_to<float>();
#pragma clang loop unroll(disable)
  for(int i=0;i<3;++i) y=fmulN<64>(y,fsubN<64>(aie::broadcast<float,64>(2.f),fmulN<64>(sv,y)));
  acc64f inv;
  inv.from_vector(y);
  const aie::vector<bfloat16,64> ih=inv.to_vector<bfloat16>(),il=aie::sub(inv,ih).to_vector<bfloat16>();
#pragma clang loop unroll(disable)
  for(int c=0;c<4;++c) {
    acc64f o;
    o.from_vector(aie::load_v<64>(work+c*64));
    const aie::vector<bfloat16,64> oh=o.to_vector<bfloat16>(),ol=aie::sub(o,oh).to_vector<bfloat16>();
    acc64f y2=aie::mul(oh,ih);
    y2=aie::mac(y2,oh,il);
    y2=aie::mac(y2,ol,ih);
    aie::store_v(q+(head*4+c)*64,y2.to_vector<bfloat16>());
  }
}

// One head's online softmax over one element: 32 keys, 4 producer cores' K and V (1024 bytes a core:
// project's head slot, K and V^T as bfp16). The matrix products are aie2p's native block floating
// point (bfp16: 8 values share an exponent) on mac_8x8_8x8T, C[8][8] += A[8][8] x B[8][8]^T: scores
// take the 8 queries' 8-dim chunks (pre-scaled by kScoreLog2) against a producer's 8 keys' chunks
// (blocks along the head dimension, as project sent them); the output takes the 8 keys'
// probabilities (C's own layout) against that producer's V chunk transposed (blocks along keys), and
// the row sums come from the same probabilities against ones. After the head's last element (3) its
// output replaces its queries in q. work (floats): o [chunk][8][8] at 0, row sums [8][8] at 256,
// scores [producer][8][8] at 320, the queries as bfp16 at 576 (4 chunks: 64 mantissa bytes each, then
// the exponents), the probabilities likewise at 656.
static constexpr int kLastElement=3;
inline __attribute__((noinline)) void attend(bfloat16 *q,const uint8 *e,float *work,
    float *ml,const bfloat16 *bias,int head,int el) {
  ATT_TS(0)
  aie::set_rounding(aie::rounding_mode::conv_even);
  float *s=work+320;
  int8 *qm=(int8*)(work+576),*pm=(int8*)(work+656);   // 64-byte aligned mantissas
  uint32 *qe=(uint32*)(qm+256),*pe=(uint32*)(pm+256);
  if(el==0) {
    for(int i=0;i<320;i+=32) aie::store_v(work+i,aie::zeros<float,32>());
    const bfloat16 kh=(bfloat16)kScoreLog2,kl=(bfloat16)(kScoreLog2-(float)kh);
#pragma clang loop unroll(disable)
    for(int c=0;c<4;++c) {
      const aie::vector<bfloat16,64> qc=aie::load_v<64>(q+(head*4+c)*64);
      acc64f t=aie::mul(qc,aie::broadcast<bfloat16,64>(kh));
      store_bfp(qm+c*64,qe+c*2,bfp64(aie::mac(t,qc,aie::broadcast<bfloat16,64>(kl))));
    }
  }
  ATT_TS(1)
  // The running maximum is taken over the scores rounded to bf16 (32 lanes an operation), at most
  // half a bf16 step below the true one: the exponentials stay below 2^0.5 of their bound.
  aie::vector<bfloat16,64> maxv=aie::broadcast<bfloat16,64>((bfloat16)-1e30f);
  {
    // Four accumulator chains (the producers) over the 4 head-dimension chunks.
    const bfloat16 *bb=bias+el*256;
    acc64f s0,s1,s2,s3;
    s0.from_vector(aie::load_v<64>(bb));s1.from_vector(aie::load_v<64>(bb+64));
    s2.from_vector(aie::load_v<64>(bb+128));s3.from_vector(aie::load_v<64>(bb+192));
#pragma clang loop unroll(disable)
    for(int c=0;c<4;++c) {
      const v64bfp16ebs8 qc=load_bfp(qm+c*64,qe+c*2);
      const int8 *km=(const int8*)e+c*64;
      const uint32 *ke=(const uint32*)(e+512)+c*2;
      s0=acc64f(::mac_8x8_8x8T(qc,load_bfp(km,ke),(v64accfloat)s0));
      s1=acc64f(::mac_8x8_8x8T(qc,load_bfp(km+1024,ke+256),(v64accfloat)s1));
      s2=acc64f(::mac_8x8_8x8T(qc,load_bfp(km+2048,ke+512),(v64accfloat)s2));
      s3=acc64f(::mac_8x8_8x8T(qc,load_bfp(km+3072,ke+768),(v64accfloat)s3));
    }
    aie::store_v(s,s0.to_vector<float>());aie::store_v(s+64,s1.to_vector<float>());
    aie::store_v(s+128,s2.to_vector<float>());aie::store_v(s+192,s3.to_vector<float>());
    maxv=aie::max(aie::max(maxv,s0.to_vector<bfloat16>()),s1.to_vector<bfloat16>());
    maxv=aie::max(aie::max(maxv,s2.to_vector<bfloat16>()),s3.to_vector<bfloat16>());
  }
  ATT_TS(2)
  // Lazy rescaling: later elements keep the head's maximum so far unless their own is more than 64
  // above (2^64 fits fp32 and the bfp16 exponents), so the earlier sums need rescaling only then.
  // The maximum is kept as an integer (int32 in ml): the bf16 maximum, rounded.
  const int32 prev=((const int32*)ml)[0];
  int32 next=::bfloat16_to_int(aie::broadcast<bfloat16,16>(aie::reduce_max(maxv)),0)[0];
  const bool rescale=el && next>prev+64;
  if(el && !rescale) next=prev;
  ATT_TS(3)
#pragma clang loop unroll(disable)
  for(int kb=0;kb<4;++kb) {
    acc64f x;
    x.from_vector(aie::load_v<64>(s+kb*64));
    store_bfp(pm+kb*64,pe+kb*2,bfp64(exp2_64(x,next)));
  }
  ATT_TS(4)
  if(rescale) {
    // The earlier elements' sums, rescaled to this element's maximum: by 2^(prev - next), exact.
    const int32 be=127+prev-next;
    const v32f rb=aie::broadcast<int32,32>((be<1 ? 1 : be)<<23).cast_to<float>();
    v32b rbh,rbl;     // fmulN with rb split once
    split32(rb,rbh,rbl);
#pragma clang loop unroll(disable)
    for(int i=0;i<320;i+=32) {
      v32b wh,wl;
      split32(aie::load_v<32>(work+i),wh,wl);
      accf32 acc=aie::mul(wh,rbh);
      acc=aie::mac(acc,wh,rbl);
      acc=aie::mac(acc,wl,rbh);
      aie::store_v(work+i,acc.to_vector<float>());
    }
  }
  ((int32*)ml)[0]=next;
  ATT_TS(5)
  {
    acc64f os;
    os.from_vector(aie::load_v<64>(work+256));
    const acc64f one=ups64(aie::broadcast<bfloat16,64>(1.f));
#pragma clang loop unroll(disable)
    for(int kb=0;kb<4;++kb) os=acc64f(::mac_8x8_8x8T(load_bfp(pm+kb*64,pe+kb*2),bfp64(one),(v64accfloat)os));
    aie::store_v(work+256,os.to_vector<float>());
  }
  {
    // The four output chunks: four accumulator chains over the 4 producers.
    acc64f o0,o1,o2,o3;
    o0.from_vector(aie::load_v<64>(work));o1.from_vector(aie::load_v<64>(work+64));
    o2.from_vector(aie::load_v<64>(work+128));o3.from_vector(aie::load_v<64>(work+192));
#pragma clang loop unroll(disable)
    for(int kb=0;kb<4;++kb) {
      // V^T: producer kb's chunks, [8 dims][8 tokens] (blocks along the keys).
      const int8 *vm=(const int8*)(e+kb*1024+256);
      const uint32 *ve=(const uint32*)(e+kb*1024+544);
      const v64bfp16ebs8 p=load_bfp(pm+kb*64,pe+kb*2);
#ifndef F2_PROF_NOMM
      o0=acc64f(::mac_8x8_8x8T(p,load_bfp(vm,ve),(v64accfloat)o0));
      o1=acc64f(::mac_8x8_8x8T(p,load_bfp(vm+64,ve+2),(v64accfloat)o1));
      o2=acc64f(::mac_8x8_8x8T(p,load_bfp(vm+128,ve+4),(v64accfloat)o2));
      o3=acc64f(::mac_8x8_8x8T(p,load_bfp(vm+192,ve+6),(v64accfloat)o3));
#endif
    }
    aie::store_v(work,o0.to_vector<float>());aie::store_v(work+64,o1.to_vector<float>());
    aie::store_v(work+128,o2.to_vector<float>());aie::store_v(work+192,o3.to_vector<float>());
  }
  ATT_TS(6)
  if(el==kLastElement) finish_attention(work,q,head);
  ATT_TS(7)
}
