
#pragma once
#ifndef KERNEL_COL_H
#define KERNEL_COL_H

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


struct exo_win_1i16{
    int16_t * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1i16c{
    const int16_t * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1i32{
    int32_t * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1i32c{
    const int32_t * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1i8{
    int8_t * const data;
    const int_fast32_t strides[1];
};
struct exo_win_1i8c{
    const int8_t * const data;
    const int_fast32_t strides[1];
};
struct exo_win_2i16c{
    const int16_t * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2i32{
    int32_t * const data;
    const int_fast32_t strides[2];
};
struct exo_win_2i8c{
    const int8_t * const data;
    const int_fast32_t strides[2];
};
// gemm_NEON_10x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 10] @DRAM
// )
void gemm_NEON_10x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 10] @DRAM
// )
void gemm_NEON_10x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 10] @DRAM
// )
void gemm_NEON_10x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 10] @DRAM
// )
void gemm_NEON_10x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 10] @DRAM
// )
void gemm_NEON_10x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 10] @DRAM
// )
void gemm_NEON_10x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 10] @DRAM
// )
void gemm_NEON_10x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 10] @DRAM
// )
void gemm_NEON_10x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 11] @DRAM
// )
void gemm_NEON_11x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 11] @DRAM
// )
void gemm_NEON_11x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 11] @DRAM
// )
void gemm_NEON_11x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 11] @DRAM
// )
void gemm_NEON_11x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 11] @DRAM
// )
void gemm_NEON_11x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 11] @DRAM
// )
void gemm_NEON_11x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 11] @DRAM
// )
void gemm_NEON_11x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 11] @DRAM
// )
void gemm_NEON_11x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 12] @DRAM
// )
void gemm_NEON_12x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 12] @DRAM
// )
void gemm_NEON_12x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 12] @DRAM
// )
void gemm_NEON_12x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 12] @DRAM
// )
void gemm_NEON_12x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 12] @DRAM
// )
void gemm_NEON_12x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 12] @DRAM
// )
void gemm_NEON_12x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 12] @DRAM
// )
void gemm_NEON_12x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 12] @DRAM
// )
void gemm_NEON_12x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_1x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 1] @DRAM
// )
void gemm_NEON_1x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 1] @DRAM
// )
void gemm_NEON_1x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 2] @DRAM
// )
void gemm_NEON_2x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 2] @DRAM
// )
void gemm_NEON_2x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 3] @DRAM
// )
void gemm_NEON_3x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 3] @DRAM
// )
void gemm_NEON_3x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_5x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 5] @DRAM
// )
void gemm_NEON_5x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 5] @DRAM
// )
void gemm_NEON_5x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 5] @DRAM
// )
void gemm_NEON_5x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 5] @DRAM
// )
void gemm_NEON_5x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 5] @DRAM
// )
void gemm_NEON_5x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 5] @DRAM
// )
void gemm_NEON_5x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 5] @DRAM
// )
void gemm_NEON_5x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 5] @DRAM
// )
void gemm_NEON_5x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 6] @DRAM
// )
void gemm_NEON_6x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 6] @DRAM
// )
void gemm_NEON_6x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 6] @DRAM
// )
void gemm_NEON_6x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 6] @DRAM
// )
void gemm_NEON_6x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 6] @DRAM
// )
void gemm_NEON_6x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 6] @DRAM
// )
void gemm_NEON_6x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 6] @DRAM
// )
void gemm_NEON_6x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 6] @DRAM
// )
void gemm_NEON_6x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 7] @DRAM
// )
void gemm_NEON_7x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 7] @DRAM
// )
void gemm_NEON_7x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 7] @DRAM
// )
void gemm_NEON_7x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 7] @DRAM
// )
void gemm_NEON_7x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 7] @DRAM
// )
void gemm_NEON_7x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 7] @DRAM
// )
void gemm_NEON_7x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 7] @DRAM
// )
void gemm_NEON_7x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 7] @DRAM
// )
void gemm_NEON_7x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 8] @DRAM
// )
void gemm_NEON_8x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 8] @DRAM
// )
void gemm_NEON_8x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 8] @DRAM
// )
void gemm_NEON_8x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 8] @DRAM
// )
void gemm_NEON_8x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 8] @DRAM
// )
void gemm_NEON_8x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 8] @DRAM
// )
void gemm_NEON_8x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 8] @DRAM
// )
void gemm_NEON_8x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 8] @DRAM
// )
void gemm_NEON_8x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_9x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 9] @DRAM
// )
void gemm_NEON_9x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 9] @DRAM
// )
void gemm_NEON_9x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 9] @DRAM
// )
void gemm_NEON_9x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 9] @DRAM
// )
void gemm_NEON_9x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 9] @DRAM
// )
void gemm_NEON_9x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 9] @DRAM
// )
void gemm_NEON_9x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 9] @DRAM
// )
void gemm_NEON_9x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 9] @DRAM
// )
void gemm_NEON_9x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );



#ifdef __cplusplus
}
#endif
#endif  // KERNEL_COL_H
