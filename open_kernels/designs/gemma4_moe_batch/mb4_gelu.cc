// h band `band` = gelu_tanh(g) * u, into one 1024 B output element
#include "mb4.h"
extern "C" {
void mb_gelu(const float *__restrict u, const float *__restrict g, float *__restrict h, int band) {
  mb_gelu_band(u + band * 64 * MB_NT, g + band * 64 * MB_NT, reinterpret_cast<bfloat16 *>(h));
}
}
