// Add the learned router bias, choose stable top four, softmax those logits.
#include "vecmath.h"
#include <cstdint>

extern "C" void gptoss_router_fin(const float *__restrict acc,
                                   const bfloat16 *__restrict x_bias,
                                   uint8_t *__restrict out) {
  const bfloat16 *bias = x_bias + 2880;
  float *logits = reinterpret_cast<float *>(out + 32);
  int32_t *ids = reinterpret_cast<int32_t *>(out);
  float *weights = reinterpret_cast<float *>(out + 16);
  for (unsigned e = 0; e < 32; ++e) logits[e] = acc[e] + (float)bias[e];
  uint32_t used = 0;
  float selected[4];
  for (unsigned k = 0; k < 4; ++k) {
    int32_t best_id = -1;
    float best = -3.402823466e38f;
    for (unsigned e = 0; e < 32; ++e) {
      if (used & (1u << e)) continue;
      if (best_id < 0 || logits[e] > best) { best = logits[e]; best_id = (int32_t)e; }
    }
    ids[k] = best_id;
    selected[k] = best;
    used |= 1u << best_id;
  }
  const float m = selected[0];
  float sum = 0.f;
  for (unsigned k = 0; k < 4; ++k) {
    weights[k] = sexp(selected[k] - m);
    sum += weights[k];
  }
  for (unsigned k = 0; k < 4; ++k) weights[k] /= sum;
}
