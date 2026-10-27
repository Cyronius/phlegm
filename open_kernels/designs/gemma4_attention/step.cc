// Online softmax over one normalized, rotated BF16 Q/K and normalized BF16 V.
// Gemma 4 text attention uses score scaling 1.0.
#include "vecmath.h"
#include <cstdint>

#ifndef HEAD_DIM
#error HEAD_DIM must be defined by the IRON specialization
#endif

extern "C" void gemma4_attention_step(const uint8_t *__restrict meta,
                                       const bfloat16 *__restrict kv,
                                       float *__restrict state) {
  const bfloat16 *query = reinterpret_cast<const bfloat16 *>(meta);
  float score = 0.0f;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < HEAD_DIM; ++j)
    score += (float)query[j] * (float)kv[j];
  const float old_max = state[HEAD_DIM];
  const float next_max = score > old_max ? score : old_max;
  const float old_mass = sexp(old_max - next_max);
  const float new_mass = sexp(score - next_max);
  state[HEAD_DIM] = next_max;
  state[HEAD_DIM + 1] = state[HEAD_DIM + 1] * old_mass + new_mass;
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < HEAD_DIM; ++j)
    state[j] = state[j] * old_mass + new_mass * (float)kv[HEAD_DIM + j];
}
