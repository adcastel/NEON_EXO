
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
// gemm_NEON_1x10_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x10_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x11_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x11_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x12_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x12_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x13_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 1] @DRAM
// )
void gemm_NEON_1x13_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x13_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 1] @DRAM
// )
void gemm_NEON_1x13_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x14_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 1] @DRAM
// )
void gemm_NEON_1x14_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x14_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 1] @DRAM
// )
void gemm_NEON_1x14_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x15_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 1] @DRAM
// )
void gemm_NEON_1x15_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x15_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 1] @DRAM
// )
void gemm_NEON_1x15_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x16_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 1] @DRAM
// )
void gemm_NEON_1x16_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x16_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 1] @DRAM
// )
void gemm_NEON_1x16_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x17_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 1] @DRAM
// )
void gemm_NEON_1x17_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x17_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 1] @DRAM
// )
void gemm_NEON_1x17_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x18_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 1] @DRAM
// )
void gemm_NEON_1x18_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x18_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 1] @DRAM
// )
void gemm_NEON_1x18_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x19_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 1] @DRAM
// )
void gemm_NEON_1x19_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x19_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 1] @DRAM
// )
void gemm_NEON_1x19_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x20_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 1] @DRAM
// )
void gemm_NEON_1x20_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x20_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 1] @DRAM
// )
void gemm_NEON_1x20_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 1] @DRAM
// )
void gemm_NEON_1x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 1] @DRAM
// )
void gemm_NEON_1x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x5_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x5_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x6_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x6_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x7_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x7_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x8_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x8_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x9_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x9_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x10_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x10_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x11_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x11_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x12_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x12_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x13_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 2] @DRAM
// )
void gemm_NEON_2x13_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x13_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 2] @DRAM
// )
void gemm_NEON_2x13_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x14_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 2] @DRAM
// )
void gemm_NEON_2x14_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x14_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 2] @DRAM
// )
void gemm_NEON_2x14_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x15_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 2] @DRAM
// )
void gemm_NEON_2x15_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x15_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 2] @DRAM
// )
void gemm_NEON_2x15_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x16_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 2] @DRAM
// )
void gemm_NEON_2x16_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x16_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 2] @DRAM
// )
void gemm_NEON_2x16_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x17_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 2] @DRAM
// )
void gemm_NEON_2x17_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x17_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 2] @DRAM
// )
void gemm_NEON_2x17_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x18_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 2] @DRAM
// )
void gemm_NEON_2x18_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x18_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 2] @DRAM
// )
void gemm_NEON_2x18_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x19_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 2] @DRAM
// )
void gemm_NEON_2x19_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x19_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 2] @DRAM
// )
void gemm_NEON_2x19_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x20_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 2] @DRAM
// )
void gemm_NEON_2x20_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x20_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 2] @DRAM
// )
void gemm_NEON_2x20_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 2] @DRAM
// )
void gemm_NEON_2x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 2] @DRAM
// )
void gemm_NEON_2x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x5_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x5_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x6_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x6_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x7_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x7_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x8_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x8_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x9_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x9_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x10_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x10_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x11_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x11_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x12_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x12_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x13_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 3] @DRAM
// )
void gemm_NEON_3x13_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x13_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 3] @DRAM
// )
void gemm_NEON_3x13_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x14_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 3] @DRAM
// )
void gemm_NEON_3x14_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x14_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 3] @DRAM
// )
void gemm_NEON_3x14_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x15_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 3] @DRAM
// )
void gemm_NEON_3x15_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x15_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 3] @DRAM
// )
void gemm_NEON_3x15_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x16_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 3] @DRAM
// )
void gemm_NEON_3x16_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x16_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 3] @DRAM
// )
void gemm_NEON_3x16_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x17_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 3] @DRAM
// )
void gemm_NEON_3x17_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x17_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 3] @DRAM
// )
void gemm_NEON_3x17_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x18_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 3] @DRAM
// )
void gemm_NEON_3x18_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x18_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 3] @DRAM
// )
void gemm_NEON_3x18_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x19_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 3] @DRAM
// )
void gemm_NEON_3x19_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x19_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 3] @DRAM
// )
void gemm_NEON_3x19_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x20_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 3] @DRAM
// )
void gemm_NEON_3x20_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x20_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 3] @DRAM
// )
void gemm_NEON_3x20_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 3] @DRAM
// )
void gemm_NEON_3x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 3] @DRAM
// )
void gemm_NEON_3x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x5_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x5_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x6_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x6_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x7_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x7_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x8_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x8_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x9_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x9_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x10_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x10_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x11_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x11_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x12_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x12_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x13_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 4] @DRAM
// )
void gemm_NEON_4x13_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x13_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 4] @DRAM
// )
void gemm_NEON_4x13_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x14_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 4] @DRAM
// )
void gemm_NEON_4x14_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x14_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 4] @DRAM
// )
void gemm_NEON_4x14_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x15_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 4] @DRAM
// )
void gemm_NEON_4x15_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x15_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 4] @DRAM
// )
void gemm_NEON_4x15_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x16_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 4] @DRAM
// )
void gemm_NEON_4x16_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x16_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 4] @DRAM
// )
void gemm_NEON_4x16_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x17_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 4] @DRAM
// )
void gemm_NEON_4x17_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x17_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 4] @DRAM
// )
void gemm_NEON_4x17_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x18_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 4] @DRAM
// )
void gemm_NEON_4x18_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x18_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 4] @DRAM
// )
void gemm_NEON_4x18_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x19_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 4] @DRAM
// )
void gemm_NEON_4x19_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x19_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 4] @DRAM
// )
void gemm_NEON_4x19_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x20_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 4] @DRAM
// )
void gemm_NEON_4x20_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x20_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 4] @DRAM
// )
void gemm_NEON_4x20_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
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

// gemm_NEON_4x5_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x5_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x6_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x6_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x7_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x7_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x8_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x8_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x9_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x9_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );



#ifdef __cplusplus
}
#endif
#endif  // KERNEL_COL_H
