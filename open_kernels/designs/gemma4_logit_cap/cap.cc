// Gemma 4 final_logit_softcapping = 30.0.
#include "vecmath.h"

extern "C" void gemma4_logit_cap_block(const float *__restrict x,
                                        float *__restrict y) {
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < 256; ++i) {
    const float value = x[i];
    const float magnitude = value < 0.0f ? -value : value;
    if (magnitude >= 600.0f) {
      y[i] = value < 0.0f ? -30.0f : 30.0f;
      continue;
    }
    const float e = sexp((-2.0f / 30.0f) * magnitude);
    const float capped = 30.0f * ((1.0f - e) / (1.0f + e));
    y[i] = value < 0.0f ? -capped : capped;
  }
}
