// Experimental64x64/core Q4 prefill MAC. A:[row-pair][K8][row][hi,lo];
// B:[token8][K8][hi,lo]; C:ordinary8x8 sub-blocks, FP32 across K.
// Default retains HH,HL,LH order; LL is omitted. Explicit turbo uses HH only.
// Both modes consume the same high/low panel layout; turbo ignores low limbs.
// Precision acceptance requires independent kernel and full-model gates.
#include <aie_api/aie.hpp>
using AC = aie::accum<accfloat, 64>;

extern "C" void gqp_comp(const bfp16ebs8 *__restrict a,
                          const bfp16ebs8 *__restrict b, float *__restrict c) {
    for (unsigned rb = 0; rb < 8; rb += 2) {
        for (unsigned nb = 0; nb < 8; ++nb) {
            aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sa(a), sb(b);
            sa.seek(rb * 16);
            sb.seek(nb * 16);
            AC c0(aie::load_v<64>(c + (rb * 8 + nb) * 64));
            AC c1(aie::load_v<64>(c + ((rb + 1) * 8 + nb) * 64));
            for (unsigned kb = 0; kb < 8; ++kb) {
                auto ah0 = sa.pop(), al0 = sa.pop(), bh = sb.pop(), bl = sb.pop();
                c0 = ::mac_8x8_8x8T_conf(ah0, bh, c0, 0, 0, 0);

#ifndef COMPENSATED64_TURBO_HH
                c0 = ::mac_8x8_8x8T_conf(ah0, bl, c0, 0, 0, 0);
                c0 = ::mac_8x8_8x8T_conf(al0, bh, c0, 0, 0, 0);

#endif
                auto ah1 = sa.pop(), al1 = sa.pop();
                c1 = ::mac_8x8_8x8T_conf(ah1, bh, c1, 0, 0, 0);

#ifndef COMPENSATED64_TURBO_HH
                c1 = ::mac_8x8_8x8T_conf(ah1, bl, c1, 0, 0, 0);
                c1 = ::mac_8x8_8x8T_conf(al1, bh, c1, 0, 0, 0);
#endif
            }
            aie::store_v(c + (rb * 8 + nb) * 64, c0.to_vector<float>());
            aie::store_v(c + ((rb + 1) * 8 + nb) * 64, c1.to_vector<float>());
        }
    }
}
