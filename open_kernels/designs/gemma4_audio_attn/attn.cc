// One Gemma 4 audio conformer head: 12 queries over 24 context rows.
// Q/K include the model's per-dimension and log-base scales. rel_key contains
// 13 projected relative-position vectors. The relative shift maps index
// (context_column - query_row) into these vectors when it lies in [0,13).
#include "vecmath.h"
#include <cstdint>

static inline float cap50(float value) {
  const float magnitude = value < 0.f ? -value : value;
  if (magnitude >= 1000.f) return value < 0.f ? -50.f : 50.f;
  const float e = sexp((-2.f / 50.f) * magnitude);
  const float capped = 50.f * ((1.f - e) / (1.f + e));
  return value < 0.f ? -capped : capped;
}

extern "C" void gemma4_audio_attention_head(const uint8_t *__restrict packed,
                                              float *__restrict out) {
  const float *q = reinterpret_cast<const float *>(packed);
  const float *k = reinterpret_cast<const float *>(packed + 6144);
  const float *v = reinterpret_cast<const float *>(packed + 18432);
  const float *rel_key = reinterpret_cast<const float *>(packed + 30720);
  const uint8_t *mask = packed + 37376;
#pragma clang loop unroll(disable)
  for (unsigned row = 0; row < 12; ++row) {
    float score[24];
    float max_score = -1.0e30f;
#pragma clang loop unroll(disable)
    for (unsigned col = 0; col < 24; ++col) {
      float dot = 0.f;
#pragma clang loop unroll(disable)
      for (unsigned lane = 0; lane < 128; ++lane)
        dot += q[row * 128 + lane] * k[col * 128 + lane];
      const int relative_index = (int)col - (int)row;
      if (relative_index >= 0 && relative_index < 13) {
#pragma clang loop unroll(disable)
        for (unsigned lane = 0; lane < 128; ++lane)
          dot += q[row * 128 + lane] * rel_key[relative_index * 128 + lane];
      }
      const float value = mask[row * 24 + col] ? -1.0e9f : cap50(dot);
      score[col] = value;
      if (value > max_score) max_score = value;
    }
    float total = 0.f;
#pragma clang loop unroll(disable)
    for (unsigned col = 0; col < 24; ++col) {
      const float probability = mask[row * 24 + col] ? 0.f :
                                sexp(score[col] - max_score);
      score[col] = probability;
      total += probability;
    }
    const float inv = total > 0.f ? 1.f / total : 0.f;
#pragma clang loop unroll(disable)
    for (unsigned lane = 0; lane < 128; ++lane) {
      float sum = 0.f;
#pragma clang loop unroll(disable)
      for (unsigned col = 0; col < 24; ++col)
        sum += score[col] * v[col * 128 + lane];
      out[row * 128 + lane] = sum * inv;
    }
  }
}
