// Export one head's accumulator, max and denominator; pad to 68 floats.
#include <cstdint>

extern "C" void gptoss_attn_state_finish(const float *__restrict acc,
                                          const float *__restrict mx,
                                          const float *__restrict den,
                                          float *__restrict out, int32_t h) {
  const unsigned head = (unsigned)h;
  for (unsigned j = 0; j < 64; ++j) out[j] = acc[head * 64 + j];
  out[64] = mx[head];
  out[65] = den[head];
  out[66] = 0.0f;
  out[67] = 0.0f;
}
