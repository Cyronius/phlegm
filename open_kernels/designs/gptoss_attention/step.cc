// One GPT-OSS GQA K/V row over 64 query heads and eight KV heads.
#include "vecmath.h"

extern "C" void gptoss_attn_step(const uint8_t *__restrict q_sink,
                                 const bfloat16 *__restrict kv,
                                 float *__restrict acc, float *__restrict mx,
                                 float *__restrict den) {
  const bfloat16 *q = reinterpret_cast<const bfloat16 *>(q_sink);
  const bfloat16 *k = kv;
  const bfloat16 *v = kv + 8 * 64;
#pragma clang loop unroll(disable)
  for (unsigned h = 0; h < 64; ++h) {
    const unsigned kvh = h >> 3;
    float dot = 0.0f;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 64; ++j)
      dot += (float)q[h * 64 + j] * (float)k[kvh * 64 + j];
    const float score = dot * 0.125f;
    const float next = score > mx[h] ? score : mx[h];
    const float old_mass = sexp(mx[h] - next);
    const float new_mass = sexp(score - next);
    mx[h] = next;
    den[h] = den[h] * old_mass + new_mass;
#pragma clang loop unroll(disable)
    for (unsigned j = 0; j < 64; ++j) {
      const unsigned off = h * 64 + j;
      acc[off] = acc[off] * old_mass + new_mass * (float)v[kvh * 64 + j];
    }
  }
}
