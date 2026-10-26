// Explicit turbo producer: one BFP high panel, no residual or low storage.
#pragma once
#include <aie_api/aie.hpp>
template<class Stream>
static inline __attribute__((always_inline)) void gqp_probe_emit(aie::vector<bfloat16,64> v, Stream &out) {
  out.push(aie::accum<accfloat,64>(v).to_vector<bfp16ebs8>());
}
