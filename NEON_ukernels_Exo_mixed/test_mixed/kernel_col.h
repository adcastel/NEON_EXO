
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
// gemm_NEON_10x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 10] @DRAM
// )
void gemm_NEON_10x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 10] @DRAM
// )
void gemm_NEON_10x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 10] @DRAM
// )
void gemm_NEON_10x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 10] @DRAM
// )
void gemm_NEON_10x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 10] @DRAM
// )
void gemm_NEON_10x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 10] @DRAM
// )
void gemm_NEON_10x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 10] @DRAM
// )
void gemm_NEON_10x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 10] @DRAM
// )
void gemm_NEON_10x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 10] @DRAM
// )
void gemm_NEON_10x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 10] @DRAM
// )
void gemm_NEON_10x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 10] @DRAM
// )
void gemm_NEON_10x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 10] @DRAM
// )
void gemm_NEON_10x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 10] @DRAM
// )
void gemm_NEON_10x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 10] @DRAM
// )
void gemm_NEON_10x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 10] @DRAM
// )
void gemm_NEON_10x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 10] @DRAM
// )
void gemm_NEON_10x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 10] @DRAM
// )
void gemm_NEON_10x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 10] @DRAM
// )
void gemm_NEON_10x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 10] @DRAM
// )
void gemm_NEON_10x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 10] @DRAM
// )
void gemm_NEON_10x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 10] @DRAM
// )
void gemm_NEON_10x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 10] @DRAM
// )
void gemm_NEON_10x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 10] @DRAM
// )
void gemm_NEON_10x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 10] @DRAM
// )
void gemm_NEON_10x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 10] @DRAM
// )
void gemm_NEON_10x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 10] @DRAM
// )
void gemm_NEON_10x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 10] @DRAM
// )
void gemm_NEON_10x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 10] @DRAM
// )
void gemm_NEON_10x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 10] @DRAM
// )
void gemm_NEON_10x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 10] @DRAM
// )
void gemm_NEON_10x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 10] @DRAM
// )
void gemm_NEON_10x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 10] @DRAM
// )
void gemm_NEON_10x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 10] @DRAM
// )
void gemm_NEON_10x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 10] @DRAM
// )
void gemm_NEON_10x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 10] @DRAM
// )
void gemm_NEON_10x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 10] @DRAM
// )
void gemm_NEON_10x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 10] @DRAM
// )
void gemm_NEON_10x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 10] @DRAM
// )
void gemm_NEON_10x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 10] @DRAM
// )
void gemm_NEON_10x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 10] @DRAM
// )
void gemm_NEON_10x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 10] @DRAM
// )
void gemm_NEON_10x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 10] @DRAM
// )
void gemm_NEON_10x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 10] @DRAM
// )
void gemm_NEON_10x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 10] @DRAM
// )
void gemm_NEON_10x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 10] @DRAM
// )
void gemm_NEON_10x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 10] @DRAM
// )
void gemm_NEON_10x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 10] @DRAM
// )
void gemm_NEON_10x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_10x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 10] @DRAM
// )
void gemm_NEON_10x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 11] @DRAM
// )
void gemm_NEON_11x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 11] @DRAM
// )
void gemm_NEON_11x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 11] @DRAM
// )
void gemm_NEON_11x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 11] @DRAM
// )
void gemm_NEON_11x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 11] @DRAM
// )
void gemm_NEON_11x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 11] @DRAM
// )
void gemm_NEON_11x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 11] @DRAM
// )
void gemm_NEON_11x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 11] @DRAM
// )
void gemm_NEON_11x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 11] @DRAM
// )
void gemm_NEON_11x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 11] @DRAM
// )
void gemm_NEON_11x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 11] @DRAM
// )
void gemm_NEON_11x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 11] @DRAM
// )
void gemm_NEON_11x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 11] @DRAM
// )
void gemm_NEON_11x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 11] @DRAM
// )
void gemm_NEON_11x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 11] @DRAM
// )
void gemm_NEON_11x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 11] @DRAM
// )
void gemm_NEON_11x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 11] @DRAM
// )
void gemm_NEON_11x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 11] @DRAM
// )
void gemm_NEON_11x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 11] @DRAM
// )
void gemm_NEON_11x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 11] @DRAM
// )
void gemm_NEON_11x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 11] @DRAM
// )
void gemm_NEON_11x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 11] @DRAM
// )
void gemm_NEON_11x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 11] @DRAM
// )
void gemm_NEON_11x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 11] @DRAM
// )
void gemm_NEON_11x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 11] @DRAM
// )
void gemm_NEON_11x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 11] @DRAM
// )
void gemm_NEON_11x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 11] @DRAM
// )
void gemm_NEON_11x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 11] @DRAM
// )
void gemm_NEON_11x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 11] @DRAM
// )
void gemm_NEON_11x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 11] @DRAM
// )
void gemm_NEON_11x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 11] @DRAM
// )
void gemm_NEON_11x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 11] @DRAM
// )
void gemm_NEON_11x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 11] @DRAM
// )
void gemm_NEON_11x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 11] @DRAM
// )
void gemm_NEON_11x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 11] @DRAM
// )
void gemm_NEON_11x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 11] @DRAM
// )
void gemm_NEON_11x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 11] @DRAM
// )
void gemm_NEON_11x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 11] @DRAM
// )
void gemm_NEON_11x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 11] @DRAM
// )
void gemm_NEON_11x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 11] @DRAM
// )
void gemm_NEON_11x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 11] @DRAM
// )
void gemm_NEON_11x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 11] @DRAM
// )
void gemm_NEON_11x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 11] @DRAM
// )
void gemm_NEON_11x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 11] @DRAM
// )
void gemm_NEON_11x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 11] @DRAM
// )
void gemm_NEON_11x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 11] @DRAM
// )
void gemm_NEON_11x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 11] @DRAM
// )
void gemm_NEON_11x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_11x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 11] @DRAM
// )
void gemm_NEON_11x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 1] @DRAM
// )
void gemm_NEON_1x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 1] @DRAM
// )
void gemm_NEON_1x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 1] @DRAM
// )
void gemm_NEON_1x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 1] @DRAM
// )
void gemm_NEON_1x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 1] @DRAM
// )
void gemm_NEON_1x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 1] @DRAM
// )
void gemm_NEON_1x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 1] @DRAM
// )
void gemm_NEON_1x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 1] @DRAM
// )
void gemm_NEON_1x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 1] @DRAM
// )
void gemm_NEON_1x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 1] @DRAM
// )
void gemm_NEON_1x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 1] @DRAM
// )
void gemm_NEON_1x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 1] @DRAM
// )
void gemm_NEON_1x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 1] @DRAM
// )
void gemm_NEON_1x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 1] @DRAM
// )
void gemm_NEON_1x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 1] @DRAM
// )
void gemm_NEON_1x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 1] @DRAM
// )
void gemm_NEON_1x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 1] @DRAM
// )
void gemm_NEON_1x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 1] @DRAM
// )
void gemm_NEON_1x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 1] @DRAM
// )
void gemm_NEON_1x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 1] @DRAM
// )
void gemm_NEON_1x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 1] @DRAM
// )
void gemm_NEON_1x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 1] @DRAM
// )
void gemm_NEON_1x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 1] @DRAM
// )
void gemm_NEON_1x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 1] @DRAM
// )
void gemm_NEON_1x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 1] @DRAM
// )
void gemm_NEON_1x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 1] @DRAM
// )
void gemm_NEON_1x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_1x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 2] @DRAM
// )
void gemm_NEON_2x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 2] @DRAM
// )
void gemm_NEON_2x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 2] @DRAM
// )
void gemm_NEON_2x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 2] @DRAM
// )
void gemm_NEON_2x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 2] @DRAM
// )
void gemm_NEON_2x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 2] @DRAM
// )
void gemm_NEON_2x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 2] @DRAM
// )
void gemm_NEON_2x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 2] @DRAM
// )
void gemm_NEON_2x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 2] @DRAM
// )
void gemm_NEON_2x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 2] @DRAM
// )
void gemm_NEON_2x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 2] @DRAM
// )
void gemm_NEON_2x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 2] @DRAM
// )
void gemm_NEON_2x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 2] @DRAM
// )
void gemm_NEON_2x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 2] @DRAM
// )
void gemm_NEON_2x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 2] @DRAM
// )
void gemm_NEON_2x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 2] @DRAM
// )
void gemm_NEON_2x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 2] @DRAM
// )
void gemm_NEON_2x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 2] @DRAM
// )
void gemm_NEON_2x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 2] @DRAM
// )
void gemm_NEON_2x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 2] @DRAM
// )
void gemm_NEON_2x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 2] @DRAM
// )
void gemm_NEON_2x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 2] @DRAM
// )
void gemm_NEON_2x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 2] @DRAM
// )
void gemm_NEON_2x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 2] @DRAM
// )
void gemm_NEON_2x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 2] @DRAM
// )
void gemm_NEON_2x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 2] @DRAM
// )
void gemm_NEON_2x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_2x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 3] @DRAM
// )
void gemm_NEON_3x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 3] @DRAM
// )
void gemm_NEON_3x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 3] @DRAM
// )
void gemm_NEON_3x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 3] @DRAM
// )
void gemm_NEON_3x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 3] @DRAM
// )
void gemm_NEON_3x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 3] @DRAM
// )
void gemm_NEON_3x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 3] @DRAM
// )
void gemm_NEON_3x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 3] @DRAM
// )
void gemm_NEON_3x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 3] @DRAM
// )
void gemm_NEON_3x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 3] @DRAM
// )
void gemm_NEON_3x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 3] @DRAM
// )
void gemm_NEON_3x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 3] @DRAM
// )
void gemm_NEON_3x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 3] @DRAM
// )
void gemm_NEON_3x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 3] @DRAM
// )
void gemm_NEON_3x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 3] @DRAM
// )
void gemm_NEON_3x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 3] @DRAM
// )
void gemm_NEON_3x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 3] @DRAM
// )
void gemm_NEON_3x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 3] @DRAM
// )
void gemm_NEON_3x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 3] @DRAM
// )
void gemm_NEON_3x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 3] @DRAM
// )
void gemm_NEON_3x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 3] @DRAM
// )
void gemm_NEON_3x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 3] @DRAM
// )
void gemm_NEON_3x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 3] @DRAM
// )
void gemm_NEON_3x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 3] @DRAM
// )
void gemm_NEON_3x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 3] @DRAM
// )
void gemm_NEON_3x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 3] @DRAM
// )
void gemm_NEON_3x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_3x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 4] @DRAM
// )
void gemm_NEON_4x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 4] @DRAM
// )
void gemm_NEON_4x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 4] @DRAM
// )
void gemm_NEON_4x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 4] @DRAM
// )
void gemm_NEON_4x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 4] @DRAM
// )
void gemm_NEON_4x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 4] @DRAM
// )
void gemm_NEON_4x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 4] @DRAM
// )
void gemm_NEON_4x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 4] @DRAM
// )
void gemm_NEON_4x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 4] @DRAM
// )
void gemm_NEON_4x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 4] @DRAM
// )
void gemm_NEON_4x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 4] @DRAM
// )
void gemm_NEON_4x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 4] @DRAM
// )
void gemm_NEON_4x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 4] @DRAM
// )
void gemm_NEON_4x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 4] @DRAM
// )
void gemm_NEON_4x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 4] @DRAM
// )
void gemm_NEON_4x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 4] @DRAM
// )
void gemm_NEON_4x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 4] @DRAM
// )
void gemm_NEON_4x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 4] @DRAM
// )
void gemm_NEON_4x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 4] @DRAM
// )
void gemm_NEON_4x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 4] @DRAM
// )
void gemm_NEON_4x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 4] @DRAM
// )
void gemm_NEON_4x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 4] @DRAM
// )
void gemm_NEON_4x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 4] @DRAM
// )
void gemm_NEON_4x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 4] @DRAM
// )
void gemm_NEON_4x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_4x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_4x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 5] @DRAM
// )
void gemm_NEON_5x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 5] @DRAM
// )
void gemm_NEON_5x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 5] @DRAM
// )
void gemm_NEON_5x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 5] @DRAM
// )
void gemm_NEON_5x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 5] @DRAM
// )
void gemm_NEON_5x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 5] @DRAM
// )
void gemm_NEON_5x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 5] @DRAM
// )
void gemm_NEON_5x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 5] @DRAM
// )
void gemm_NEON_5x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 5] @DRAM
// )
void gemm_NEON_5x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 5] @DRAM
// )
void gemm_NEON_5x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 5] @DRAM
// )
void gemm_NEON_5x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 5] @DRAM
// )
void gemm_NEON_5x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 5] @DRAM
// )
void gemm_NEON_5x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 5] @DRAM
// )
void gemm_NEON_5x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 5] @DRAM
// )
void gemm_NEON_5x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 5] @DRAM
// )
void gemm_NEON_5x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 5] @DRAM
// )
void gemm_NEON_5x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 5] @DRAM
// )
void gemm_NEON_5x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 5] @DRAM
// )
void gemm_NEON_5x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 5] @DRAM
// )
void gemm_NEON_5x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 5] @DRAM
// )
void gemm_NEON_5x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 5] @DRAM
// )
void gemm_NEON_5x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 5] @DRAM
// )
void gemm_NEON_5x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 5] @DRAM
// )
void gemm_NEON_5x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 5] @DRAM
// )
void gemm_NEON_5x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 5] @DRAM
// )
void gemm_NEON_5x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 5] @DRAM
// )
void gemm_NEON_5x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 5] @DRAM
// )
void gemm_NEON_5x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 5] @DRAM
// )
void gemm_NEON_5x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 5] @DRAM
// )
void gemm_NEON_5x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 5] @DRAM
// )
void gemm_NEON_5x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 5] @DRAM
// )
void gemm_NEON_5x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 5] @DRAM
// )
void gemm_NEON_5x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 5] @DRAM
// )
void gemm_NEON_5x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 5] @DRAM
// )
void gemm_NEON_5x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 5] @DRAM
// )
void gemm_NEON_5x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 5] @DRAM
// )
void gemm_NEON_5x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 5] @DRAM
// )
void gemm_NEON_5x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 5] @DRAM
// )
void gemm_NEON_5x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 5] @DRAM
// )
void gemm_NEON_5x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 5] @DRAM
// )
void gemm_NEON_5x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 5] @DRAM
// )
void gemm_NEON_5x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 5] @DRAM
// )
void gemm_NEON_5x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 5] @DRAM
// )
void gemm_NEON_5x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 5] @DRAM
// )
void gemm_NEON_5x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 5] @DRAM
// )
void gemm_NEON_5x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 5] @DRAM
// )
void gemm_NEON_5x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_5x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 5] @DRAM
// )
void gemm_NEON_5x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 6] @DRAM
// )
void gemm_NEON_6x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 6] @DRAM
// )
void gemm_NEON_6x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 6] @DRAM
// )
void gemm_NEON_6x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 6] @DRAM
// )
void gemm_NEON_6x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 6] @DRAM
// )
void gemm_NEON_6x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 6] @DRAM
// )
void gemm_NEON_6x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 6] @DRAM
// )
void gemm_NEON_6x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 6] @DRAM
// )
void gemm_NEON_6x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 6] @DRAM
// )
void gemm_NEON_6x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 6] @DRAM
// )
void gemm_NEON_6x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 6] @DRAM
// )
void gemm_NEON_6x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 6] @DRAM
// )
void gemm_NEON_6x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 6] @DRAM
// )
void gemm_NEON_6x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 6] @DRAM
// )
void gemm_NEON_6x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 6] @DRAM
// )
void gemm_NEON_6x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 6] @DRAM
// )
void gemm_NEON_6x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 6] @DRAM
// )
void gemm_NEON_6x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 6] @DRAM
// )
void gemm_NEON_6x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 6] @DRAM
// )
void gemm_NEON_6x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 6] @DRAM
// )
void gemm_NEON_6x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 6] @DRAM
// )
void gemm_NEON_6x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 6] @DRAM
// )
void gemm_NEON_6x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 6] @DRAM
// )
void gemm_NEON_6x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 6] @DRAM
// )
void gemm_NEON_6x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 6] @DRAM
// )
void gemm_NEON_6x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 6] @DRAM
// )
void gemm_NEON_6x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 6] @DRAM
// )
void gemm_NEON_6x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 6] @DRAM
// )
void gemm_NEON_6x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 6] @DRAM
// )
void gemm_NEON_6x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 6] @DRAM
// )
void gemm_NEON_6x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 6] @DRAM
// )
void gemm_NEON_6x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 6] @DRAM
// )
void gemm_NEON_6x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 6] @DRAM
// )
void gemm_NEON_6x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 6] @DRAM
// )
void gemm_NEON_6x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 6] @DRAM
// )
void gemm_NEON_6x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 6] @DRAM
// )
void gemm_NEON_6x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 6] @DRAM
// )
void gemm_NEON_6x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 6] @DRAM
// )
void gemm_NEON_6x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 6] @DRAM
// )
void gemm_NEON_6x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 6] @DRAM
// )
void gemm_NEON_6x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 6] @DRAM
// )
void gemm_NEON_6x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 6] @DRAM
// )
void gemm_NEON_6x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 6] @DRAM
// )
void gemm_NEON_6x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 6] @DRAM
// )
void gemm_NEON_6x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 6] @DRAM
// )
void gemm_NEON_6x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 6] @DRAM
// )
void gemm_NEON_6x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 6] @DRAM
// )
void gemm_NEON_6x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_6x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 6] @DRAM
// )
void gemm_NEON_6x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 7] @DRAM
// )
void gemm_NEON_7x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 7] @DRAM
// )
void gemm_NEON_7x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 7] @DRAM
// )
void gemm_NEON_7x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 7] @DRAM
// )
void gemm_NEON_7x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 7] @DRAM
// )
void gemm_NEON_7x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 7] @DRAM
// )
void gemm_NEON_7x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 7] @DRAM
// )
void gemm_NEON_7x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 7] @DRAM
// )
void gemm_NEON_7x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 7] @DRAM
// )
void gemm_NEON_7x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 7] @DRAM
// )
void gemm_NEON_7x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 7] @DRAM
// )
void gemm_NEON_7x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 7] @DRAM
// )
void gemm_NEON_7x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 7] @DRAM
// )
void gemm_NEON_7x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 7] @DRAM
// )
void gemm_NEON_7x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 7] @DRAM
// )
void gemm_NEON_7x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 7] @DRAM
// )
void gemm_NEON_7x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 7] @DRAM
// )
void gemm_NEON_7x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 7] @DRAM
// )
void gemm_NEON_7x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 7] @DRAM
// )
void gemm_NEON_7x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 7] @DRAM
// )
void gemm_NEON_7x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 7] @DRAM
// )
void gemm_NEON_7x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 7] @DRAM
// )
void gemm_NEON_7x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 7] @DRAM
// )
void gemm_NEON_7x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 7] @DRAM
// )
void gemm_NEON_7x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 7] @DRAM
// )
void gemm_NEON_7x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 7] @DRAM
// )
void gemm_NEON_7x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 7] @DRAM
// )
void gemm_NEON_7x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 7] @DRAM
// )
void gemm_NEON_7x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 7] @DRAM
// )
void gemm_NEON_7x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 7] @DRAM
// )
void gemm_NEON_7x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 7] @DRAM
// )
void gemm_NEON_7x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 7] @DRAM
// )
void gemm_NEON_7x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 7] @DRAM
// )
void gemm_NEON_7x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 7] @DRAM
// )
void gemm_NEON_7x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 7] @DRAM
// )
void gemm_NEON_7x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 7] @DRAM
// )
void gemm_NEON_7x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 7] @DRAM
// )
void gemm_NEON_7x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 7] @DRAM
// )
void gemm_NEON_7x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 7] @DRAM
// )
void gemm_NEON_7x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 7] @DRAM
// )
void gemm_NEON_7x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 7] @DRAM
// )
void gemm_NEON_7x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 7] @DRAM
// )
void gemm_NEON_7x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 7] @DRAM
// )
void gemm_NEON_7x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 7] @DRAM
// )
void gemm_NEON_7x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 7] @DRAM
// )
void gemm_NEON_7x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 7] @DRAM
// )
void gemm_NEON_7x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 7] @DRAM
// )
void gemm_NEON_7x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_7x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 7] @DRAM
// )
void gemm_NEON_7x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 8] @DRAM
// )
void gemm_NEON_8x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 8] @DRAM
// )
void gemm_NEON_8x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 8] @DRAM
// )
void gemm_NEON_8x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 8] @DRAM
// )
void gemm_NEON_8x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 8] @DRAM
// )
void gemm_NEON_8x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 8] @DRAM
// )
void gemm_NEON_8x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 8] @DRAM
// )
void gemm_NEON_8x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 8] @DRAM
// )
void gemm_NEON_8x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 8] @DRAM
// )
void gemm_NEON_8x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 8] @DRAM
// )
void gemm_NEON_8x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 8] @DRAM
// )
void gemm_NEON_8x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 8] @DRAM
// )
void gemm_NEON_8x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 8] @DRAM
// )
void gemm_NEON_8x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 8] @DRAM
// )
void gemm_NEON_8x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 8] @DRAM
// )
void gemm_NEON_8x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 8] @DRAM
// )
void gemm_NEON_8x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 8] @DRAM
// )
void gemm_NEON_8x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 8] @DRAM
// )
void gemm_NEON_8x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 8] @DRAM
// )
void gemm_NEON_8x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 8] @DRAM
// )
void gemm_NEON_8x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 8] @DRAM
// )
void gemm_NEON_8x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 8] @DRAM
// )
void gemm_NEON_8x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 8] @DRAM
// )
void gemm_NEON_8x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 8] @DRAM
// )
void gemm_NEON_8x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 8] @DRAM
// )
void gemm_NEON_8x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 8] @DRAM
// )
void gemm_NEON_8x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 8] @DRAM
// )
void gemm_NEON_8x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 8] @DRAM
// )
void gemm_NEON_8x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 8] @DRAM
// )
void gemm_NEON_8x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 8] @DRAM
// )
void gemm_NEON_8x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 8] @DRAM
// )
void gemm_NEON_8x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 8] @DRAM
// )
void gemm_NEON_8x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 8] @DRAM
// )
void gemm_NEON_8x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 8] @DRAM
// )
void gemm_NEON_8x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 8] @DRAM
// )
void gemm_NEON_8x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 8] @DRAM
// )
void gemm_NEON_8x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 8] @DRAM
// )
void gemm_NEON_8x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 8] @DRAM
// )
void gemm_NEON_8x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 8] @DRAM
// )
void gemm_NEON_8x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 8] @DRAM
// )
void gemm_NEON_8x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 8] @DRAM
// )
void gemm_NEON_8x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 8] @DRAM
// )
void gemm_NEON_8x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 8] @DRAM
// )
void gemm_NEON_8x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 8] @DRAM
// )
void gemm_NEON_8x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 8] @DRAM
// )
void gemm_NEON_8x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 8] @DRAM
// )
void gemm_NEON_8x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_8x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 8] @DRAM
// )
void gemm_NEON_8x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_8x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 8] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 8] @DRAM
// )
void gemm_NEON_8x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 9] @DRAM
// )
void gemm_NEON_9x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 9] @DRAM
// )
void gemm_NEON_9x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 9] @DRAM
// )
void gemm_NEON_9x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 9] @DRAM
// )
void gemm_NEON_9x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 9] @DRAM
// )
void gemm_NEON_9x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 9] @DRAM
// )
void gemm_NEON_9x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 9] @DRAM
// )
void gemm_NEON_9x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 9] @DRAM
// )
void gemm_NEON_9x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 9] @DRAM
// )
void gemm_NEON_9x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 9] @DRAM
// )
void gemm_NEON_9x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 9] @DRAM
// )
void gemm_NEON_9x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 9] @DRAM
// )
void gemm_NEON_9x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 9] @DRAM
// )
void gemm_NEON_9x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 9] @DRAM
// )
void gemm_NEON_9x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 9] @DRAM
// )
void gemm_NEON_9x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 9] @DRAM
// )
void gemm_NEON_9x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 9] @DRAM
// )
void gemm_NEON_9x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 9] @DRAM
// )
void gemm_NEON_9x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 9] @DRAM
// )
void gemm_NEON_9x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 9] @DRAM
// )
void gemm_NEON_9x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 9] @DRAM
// )
void gemm_NEON_9x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 9] @DRAM
// )
void gemm_NEON_9x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 9] @DRAM
// )
void gemm_NEON_9x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 9] @DRAM
// )
void gemm_NEON_9x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 9] @DRAM
// )
void gemm_NEON_9x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 9] @DRAM
// )
void gemm_NEON_9x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 9] @DRAM
// )
void gemm_NEON_9x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 9] @DRAM
// )
void gemm_NEON_9x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 9] @DRAM
// )
void gemm_NEON_9x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 9] @DRAM
// )
void gemm_NEON_9x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 9] @DRAM
// )
void gemm_NEON_9x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 9] @DRAM
// )
void gemm_NEON_9x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 9] @DRAM
// )
void gemm_NEON_9x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 9] @DRAM
// )
void gemm_NEON_9x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 9] @DRAM
// )
void gemm_NEON_9x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 9] @DRAM
// )
void gemm_NEON_9x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 9] @DRAM
// )
void gemm_NEON_9x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 9] @DRAM
// )
void gemm_NEON_9x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 9] @DRAM
// )
void gemm_NEON_9x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 9] @DRAM
// )
void gemm_NEON_9x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 9] @DRAM
// )
void gemm_NEON_9x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 9] @DRAM
// )
void gemm_NEON_9x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 9] @DRAM
// )
void gemm_NEON_9x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 9] @DRAM
// )
void gemm_NEON_9x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 9] @DRAM
// )
void gemm_NEON_9x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 9] @DRAM
// )
void gemm_NEON_9x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 9] @DRAM
// )
void gemm_NEON_9x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_9x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 9] @DRAM
// )
void gemm_NEON_9x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );



#ifdef __cplusplus
}
#endif
#endif  // KERNEL_COL_H
