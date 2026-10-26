// Turbo HH-only MAC. Compact local A:[row-pair][K8][row][high].
// External B is hi/lo; explicit HIGH_B selects compact local highs after DMA gather.
// FP32 output and accumulation order match the previous HH-only variant.
#include <aie_api/aie.hpp>
using AC = aie::accum<accfloat,64>;
extern "C" void gqp_comp(const bfp16ebs8 *__restrict a,
                         const bfp16ebs8 *__restrict b, float *__restrict c) {
    for (unsigned rb = 0; rb < 8; rb += 2) {
        for (unsigned nb = 0; nb < 8; ++nb) {
            aie::block_vector_input_buffer_stream<bfp16ebs8,64> sa(a), sb(b);
            sa.seek(rb * 8);
#ifdef COMPENSATED64_HIGH_B
            sb.seek(nb * 8);
#else
            sb.seek(nb * 16);
#endif
            AC c0(aie::load_v<64>(c + (rb * 8 + nb) * 64));
            AC c1(aie::load_v<64>(c + ((rb + 1) * 8 + nb) * 64));
            for (unsigned kb = 0; kb < 8; ++kb) {
                auto ah0 = sa.pop(), bh = sb.pop();
#ifndef COMPENSATED64_HIGH_B
                sb.pop(); // Advance over the unused B low panel.
#endif
                c0 = ::mac_8x8_8x8T_conf(ah0, bh, c0, 0, 0, 0);
                auto ah1 = sa.pop();
                c1 = ::mac_8x8_8x8T_conf(ah1, bh, c1, 0, 0, 0);
            }
            aie::store_v(c + (rb * 8 + nb) * 64, c0.to_vector<float>());
            aie::store_v(c + ((rb + 1) * 8 + nb) * 64, c1.to_vector<float>());
        }
    }
}
