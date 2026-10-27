// Normalize the value accumulator for one 64-value GPT-OSS query head.
#include <cstdint>

extern "C" void gptoss_attn_finish(const float *__restrict acc,
                                   const float *__restrict den,
                                   float *__restrict output, int32_t h) {
  const float inv = 1.0f / den[(unsigned)h];
  for (unsigned j = 0; j < 64; ++j)
    output[j] = acc[(unsigned)h * 64 + j] * inv;
}
