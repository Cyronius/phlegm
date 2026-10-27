// Convert 64-lane FP32 online-softmax state to the BF16 O-projection input.
#include <cstdint>

static inline bfloat16 to_bf16_rne(float value) {
  union { float f; uint32_t u; } bits = {value};
  bits.u += 0x7fffu + ((bits.u >> 16) & 1u);
  bits.u &= 0xffff0000u;
  return bfloat16(bits.f);
}

extern "C" void gptoss_attention_finalize_head(const float *__restrict state,
                                                 bfloat16 *__restrict output) {
  const float inv_denominator = 1.0f / state[65];
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < 64; ++j)
    output[j] = to_bf16_rne(state[j] * inv_denominator);
}
