#include "router.h"
// Two consecutive W elements per call, one from each of the norm helper's two input streams
// (designs/layer_x/ux.py UX_RSPLIT): elements 2 rb and 2 rb + 1 through router_acc_impl in that
// order, so acc is accumulated exactly as router_acc accumulates it over the elements one by one.
extern "C" {
void router_acc2(const bfloat16 *__restrict W0, const bfloat16 *__restrict W1, const bfloat16 *__restrict x,
                 float *__restrict acc, int rb) {
  router_acc_impl(W0, x, acc, 2 * rb);
  router_acc_impl(W1, x, acc, 2 * rb + 1);
}
}
