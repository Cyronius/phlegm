#include "mbl.h"
extern "C" {
void mbl_zero_ug(float *__restrict c) { mbl_zero_n(c, MBL_NRB_UP * 64 * MBL_NT); }
}
