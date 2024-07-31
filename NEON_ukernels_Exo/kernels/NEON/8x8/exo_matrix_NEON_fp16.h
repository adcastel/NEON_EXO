#include "kernels_NEON_8x8_fp16.h"
#include <stdlib.h>
typedef void (*ukrFunction)( void *ctxt, int_fast32_t KC, const _Float16* alpha, _Float16 * A, int lda , _Float16 * B, int ldb, const _Float16* beta, _Float16 *C, int ldc);
ukrFunction**** allocateMatrix();
void fillMatrix(ukrFunction**** matrix);
void freeMatrix(ukrFunction**** matrix);
