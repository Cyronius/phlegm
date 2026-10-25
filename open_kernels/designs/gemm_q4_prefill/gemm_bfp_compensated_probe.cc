// Experimental three-product BFP probe, not an accepted model route.
// Tile geometry: weights64x64, activations64x32, C64x32 in8x8 blocks.
// The producer packs A [row8][k8][hi,lo], B [token16][k8][token8][hi,lo].
// Each64-value panel contains8 rows of8 K values with one exponent per row.
// Three products HH+HL+LH preserve the independent operand gate on tested
// shapes; LL is omitted. This is a precision change requiring model gates.
// ONLY_PACK and ONLY_MM keep entry symbols unique in separately compiled TUs.
#include <aie_api/aie.hpp>
using BV = aie::block_vector<bfp16ebs8, 64>;
using AC = aie::accum<accfloat, 64>;
struct Split { BV hi, lo; };
static inline __attribute__((always_inline)) Split split(aie::vector<bfloat16,64> v) {
  BV hi = AC(v).to_vector<bfp16ebs8>();
  // SDK conversion uses an identity matrix product; include its cost in timing.
  auto decoded = AC(hi).to_vector<bfloat16>();
  auto residual = aie::sub(AC(v), AC(decoded)).to_vector<bfloat16>();
  return {hi, AC(residual).to_vector<bfp16ebs8>()};
}
#ifndef ONLY_MM
extern "C" void gqp_pack(const bfloat16 *src,bfp16ebs8 *dst) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::block_vector_output_buffer_stream<bfp16ebs8,64> out(dst);
  for(unsigned i=0;i<64;++i) {
    auto a=split(aie::load_v<64>(src+i*64));
    out.push(a.hi); out.push(a.lo);
  }

}
#endif
#ifndef ONLY_PACK
#ifdef ROW_PAIRS
extern "C" void gqp_comp(const bfp16ebs8 *__restrict a,const bfp16ebs8 *__restrict b,float *__restrict c) {
 for(unsigned rb=0;rb<8;rb+=2) for(unsigned nb=0;nb<4;++nb) {
  aie::block_vector_input_buffer_stream<bfp16ebs8,64> sa(a),sb(b);
  sa.seek(rb*16);sb.seek(nb*16);
  AC c0(aie::load_v<64>(c+(rb*4+nb)*64)),c1(aie::load_v<64>(c+((rb+1)*4+nb)*64));
  for(unsigned kb=0;kb<8;++kb) {
   auto ah0=sa.pop(),al0=sa.pop(),bh=sb.pop(),bl=sb.pop();
   c0=::mac_8x8_8x8T_conf(ah0,bh,c0,0,0,0);
   c0=::mac_8x8_8x8T_conf(ah0,bl,c0,0,0,0);
   c0=::mac_8x8_8x8T_conf(al0,bh,c0,0,0,0);
   auto ah1=sa.pop(),al1=sa.pop();
   c1=::mac_8x8_8x8T_conf(ah1,bh,c1,0,0,0);
   c1=::mac_8x8_8x8T_conf(ah1,bl,c1,0,0,0);
   c1=::mac_8x8_8x8T_conf(al1,bh,c1,0,0,0);
  }
  aie::store_v(c+(rb*4+nb)*64,c0.to_vector<float>());
  aie::store_v(c+((rb+1)*4+nb)*64,c1.to_vector<float>());
 }
}

#else
extern "C" void gqp_comp(const bfp16ebs8 *__restrict a,
                          const bfp16ebs8 *__restrict b,float *__restrict c) {
  for(unsigned rb=0;rb<8;++rb) {
    for(unsigned nb=0;nb<4;nb+=2) {
      aie::block_vector_input_buffer_stream<bfp16ebs8,64> sa(a),sb(b);
      sa.seek(rb*16); sb.seek(nb*16);
      AC acc0(aie::load_v<64>(c+(rb*4+nb)*64));
      AC acc1(aie::load_v<64>(c+(rb*4+nb+1)*64));
      for(unsigned kb=0;kb<8;++kb) {
        auto ah=sa.pop(),al=sa.pop();
        auto bh0=sb.pop(),bl0=sb.pop();
        acc0=::mac_8x8_8x8T_conf(ah,bh0,acc0,0,0,0);
        acc0=::mac_8x8_8x8T_conf(ah,bl0,acc0,0,0,0);
        acc0=::mac_8x8_8x8T_conf(al,bh0,acc0,0,0,0);
        auto bh1=sb.pop(),bl1=sb.pop();
        acc1=::mac_8x8_8x8T_conf(ah,bh1,acc1,0,0,0);
        acc1=::mac_8x8_8x8T_conf(ah,bl1,acc1,0,0,0);
        acc1=::mac_8x8_8x8T_conf(al,bh1,acc1,0,0,0);
      }
      aie::store_v(c+(rb*4+nb)*64,acc0.to_vector<float>());
      aie::store_v(c+(rb*4+nb+1)*64,acc1.to_vector<float>());
    }
  }
}
#endif
#endif
