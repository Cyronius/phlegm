#include "mb4.h"
extern "C" {
void mb_zero_ug(float *__restrict c) { mb_zero_n(c, MB_NRB_UP * 64 * MB_NT); }
}
