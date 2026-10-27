#include "vecmath.h"

extern "C" void gemma4_audio_dw_step(const bfloat16 *__restrict input,
                                      const bfloat16 *__restrict weight,
                                      float *__restrict output) {
#pragma clang loop unroll(disable)
  for (unsigned t = 0; t < 32; ++t) {
#pragma clang loop unroll(disable)
    for (unsigned c = 0; c < 64; ++c) {
      float sum = 0.0f;
#pragma clang loop unroll(disable)
      for (unsigned tap = 0; tap < 5; ++tap)
        sum += (float)input[(t + tap) * 64 + c] * (float)weight[tap * 64 + c];
      output[t * 64 + c] = sum;
    }
  }
}
