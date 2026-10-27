// Skip padded K/V rows in the last bounded attention invocation.
// The shared arithmetic lives in the standalone attention step source.
#include "step.cc"

extern "C" void gptoss_attn_state_step(const uint8_t *__restrict q_state,
                                        const bfloat16 *__restrict kv,
                                        float *__restrict acc, float *__restrict mx,
                                        float *__restrict den, int32_t row) {
  const int32_t *meta = reinterpret_cast<const int32_t *>(q_state + 64 * 64 * 2 + 64 * 68 * 4);
  if (row < meta[0]) gptoss_attn_step(q_state, kv, acc, mx, den);
}
