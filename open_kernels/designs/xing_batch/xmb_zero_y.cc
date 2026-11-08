#include "xmb.h"
extern "C" {
void xmb_zero_y(float *__restrict c) { xmb_zero_n(c, 64 * XMB_NT); }
}
