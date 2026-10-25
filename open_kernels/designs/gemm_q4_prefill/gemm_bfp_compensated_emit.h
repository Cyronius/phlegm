// Probe-only emitter used by build_bfp_stream_probe.py --fused.
// BF16 values are rounded to BFP high and residual panels; the MAC uses three products.
#pragma once
#include <aie_api/aie.hpp>
template<class Stream>
static inline __attribute__((always_inline)) void gqp_probe_emit(aie::vector<bfloat16,64> v,Stream &out) {
  using AC=aie::accum<accfloat,64>;
  auto hi=AC(v).to_vector<bfp16ebs8>();
  auto decoded=AC(hi).to_vector<bfloat16>();
  auto lo=aie::sub(AC(v),AC(decoded)).to_vector<bfloat16>();
  out.push(hi);out.push(AC(lo).to_vector<bfp16ebs8>());
}
