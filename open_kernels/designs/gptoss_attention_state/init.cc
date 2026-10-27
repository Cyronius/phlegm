// Restore per-head online-softmax state after the 8192-byte BF16 query.
#include <cstdint>

extern "C" void gptoss_attn_state_init(const uint8_t *__restrict q_state,
                                        float *__restrict acc, float *__restrict mx,
                                        float *__restrict den) {
  const float *state = reinterpret_cast<const float *>(q_state + 64 * 64 * 2);
  for (unsigned h = 0; h < 64; ++h) {
    const unsigned base = h * 68;
    for (unsigned j = 0; j < 64; ++j) acc[h * 64 + j] = state[base + j];
    mx[h] = state[base + 64];
    den[h] = state[base + 65];
  }
}
