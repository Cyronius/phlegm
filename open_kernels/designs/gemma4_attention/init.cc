// Copy the online softmax state. All-zero input is the empty state.
#include <cstdint>
#ifndef HEAD_DIM
#error HEAD_DIM must be defined by the IRON specialization
#endif

extern "C" void gemma4_attention_init(const uint8_t *__restrict meta,
                                       float *__restrict current) {
  const float *previous = reinterpret_cast<const float *>(meta + HEAD_DIM * 2);
#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < HEAD_DIM + 16; ++i)
    current[i] = previous[i];
}
