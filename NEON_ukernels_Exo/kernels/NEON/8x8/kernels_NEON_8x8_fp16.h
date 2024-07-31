
#pragma once
#ifndef KERNELS_NEON_8X8_FP16_H
#define KERNELS_NEON_8X8_FP16_H

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif


struct exo_win_1f16{
    _Float16 * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1f16c{
    const _Float16 * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2f16{
    _Float16 * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2f16c{
    const _Float16 * const data;
    const int_fast32_t strides[2];
};
// gemm_NEON_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_1x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_1x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_1x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_1x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_1x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_1x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_1x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_1x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_1x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_2x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_2x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_2x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_2x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_2x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_2x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_2x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_2x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_2x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_3x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_3x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_3x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_3x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_3x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_3x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_3x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_3x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_3x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_4x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_4x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_4x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_4x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_4x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_4x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_4x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_4x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_4x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_5x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_5x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_5x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_5x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_5x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_5x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_5x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_5x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_5x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_6x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_6x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_6x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_6x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_6x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_6x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_6x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_6x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_6x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_7x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_7x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_7x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_7x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_7x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_7x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_7x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_7x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_7x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_8x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_8x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_8x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_8x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_8x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_8x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci );

// gemm_NEON_8x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_NEON_8x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );

// gemm_NEON_8x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_NEON_8x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc );



#ifdef __cplusplus
}
#endif
#endif  // KERNELS_NEON_8X8_FP16_H
