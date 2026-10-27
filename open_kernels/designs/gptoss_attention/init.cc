// Seed the online softmax with the learned sink's mass and zero value.
#include <cstdint>

extern "C" void gptoss_attn_init(const uint8_t *__restrict q_sink,
                                  float *__restrict acc, float *__restrict mx,
                                  float *__restrict den) {
  const float *sinks = reinterpret_cast<const float *>(q_sink + 64 * 64 * 2);
  for (unsigned h = 0; h < 64; ++h) {
    mx[h] = sinks[h];
    den[h] = 1.0f;
  }
  for (unsigned i = 0; i < 64 * 64; ++i) acc[i] = 0.0f;
}
