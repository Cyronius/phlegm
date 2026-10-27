// Preserve the carry state and expose its normalized attention output.
#ifndef HEAD_DIM
#error HEAD_DIM must be defined by the IRON specialization
#endif

extern "C" void gemma4_attention_finish(float *__restrict state) {
  const float inv = 1.0f / state[HEAD_DIM + 1];
#pragma clang loop unroll(disable)
  for (unsigned j = 0; j < HEAD_DIM; ++j)
    state[HEAD_DIM + 16 + j] = state[j] * inv;
}
