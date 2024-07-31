#include "kernels_NEON_12x16_fp32.h"
#include <stdlib.h>
typedef void (*ukrFunction)( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda , float * B, int ldb, const float* beta, float *C, int ldc);
ukrFunction**** allocateMatrix();
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix);
