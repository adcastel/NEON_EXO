
#pragma once
#ifndef MIXED2_H
#define MIXED2_H

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
// gemm_NEON_12x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 12] @DRAM
// )
void gemm_NEON_12x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 12] @DRAM
// )
void gemm_NEON_12x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 12] @DRAM
// )
void gemm_NEON_12x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 12] @DRAM
// )
void gemm_NEON_12x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 12] @DRAM
// )
void gemm_NEON_12x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 12] @DRAM
// )
void gemm_NEON_12x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 12] @DRAM
// )
void gemm_NEON_12x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 12] @DRAM
// )
void gemm_NEON_12x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 12] @DRAM
// )
void gemm_NEON_12x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 12] @DRAM
// )
void gemm_NEON_12x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 12] @DRAM
// )
void gemm_NEON_12x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 12] @DRAM
// )
void gemm_NEON_12x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 12] @DRAM
// )
void gemm_NEON_12x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 12] @DRAM
// )
void gemm_NEON_12x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 12] @DRAM
// )
void gemm_NEON_12x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 12] @DRAM
// )
void gemm_NEON_12x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 12] @DRAM
// )
void gemm_NEON_12x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 12] @DRAM
// )
void gemm_NEON_12x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 12] @DRAM
// )
void gemm_NEON_12x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 12] @DRAM
// )
void gemm_NEON_12x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 12] @DRAM
// )
void gemm_NEON_12x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 12] @DRAM
// )
void gemm_NEON_12x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 12] @DRAM
// )
void gemm_NEON_12x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 12] @DRAM
// )
void gemm_NEON_12x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 12] @DRAM
// )
void gemm_NEON_12x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 12] @DRAM
// )
void gemm_NEON_12x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 12] @DRAM
// )
void gemm_NEON_12x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 12] @DRAM
// )
void gemm_NEON_12x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 12] @DRAM
// )
void gemm_NEON_12x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 12] @DRAM
// )
void gemm_NEON_12x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 12] @DRAM
// )
void gemm_NEON_12x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 12] @DRAM
// )
void gemm_NEON_12x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 12] @DRAM
// )
void gemm_NEON_12x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 12] @DRAM
// )
void gemm_NEON_12x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 12] @DRAM
// )
void gemm_NEON_12x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 12] @DRAM
// )
void gemm_NEON_12x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 12] @DRAM
// )
void gemm_NEON_12x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 12] @DRAM
// )
void gemm_NEON_12x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 12] @DRAM
// )
void gemm_NEON_12x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 12] @DRAM
// )
void gemm_NEON_12x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 12] @DRAM
// )
void gemm_NEON_12x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 12] @DRAM
// )
void gemm_NEON_12x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 12] @DRAM
// )
void gemm_NEON_12x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 12] @DRAM
// )
void gemm_NEON_12x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 12] @DRAM
// )
void gemm_NEON_12x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 12] @DRAM
// )
void gemm_NEON_12x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_12x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 12] @DRAM
// )
void gemm_NEON_12x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_12x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 12] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 12] @DRAM
// )
void gemm_NEON_12x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 13] @DRAM
// )
void gemm_NEON_13x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 13] @DRAM
// )
void gemm_NEON_13x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 13] @DRAM
// )
void gemm_NEON_13x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 13] @DRAM
// )
void gemm_NEON_13x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 13] @DRAM
// )
void gemm_NEON_13x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 13] @DRAM
// )
void gemm_NEON_13x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 13] @DRAM
// )
void gemm_NEON_13x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 13] @DRAM
// )
void gemm_NEON_13x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 13] @DRAM
// )
void gemm_NEON_13x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 13] @DRAM
// )
void gemm_NEON_13x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 13] @DRAM
// )
void gemm_NEON_13x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 13] @DRAM
// )
void gemm_NEON_13x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 13] @DRAM
// )
void gemm_NEON_13x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 13] @DRAM
// )
void gemm_NEON_13x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 13] @DRAM
// )
void gemm_NEON_13x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 13] @DRAM
// )
void gemm_NEON_13x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 13] @DRAM
// )
void gemm_NEON_13x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 13] @DRAM
// )
void gemm_NEON_13x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 13] @DRAM
// )
void gemm_NEON_13x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 13] @DRAM
// )
void gemm_NEON_13x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 13] @DRAM
// )
void gemm_NEON_13x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 13] @DRAM
// )
void gemm_NEON_13x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 13] @DRAM
// )
void gemm_NEON_13x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 13] @DRAM
// )
void gemm_NEON_13x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 13] @DRAM
// )
void gemm_NEON_13x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 13] @DRAM
// )
void gemm_NEON_13x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 13] @DRAM
// )
void gemm_NEON_13x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 13] @DRAM
// )
void gemm_NEON_13x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 13] @DRAM
// )
void gemm_NEON_13x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 13] @DRAM
// )
void gemm_NEON_13x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 13] @DRAM
// )
void gemm_NEON_13x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 13] @DRAM
// )
void gemm_NEON_13x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 13] @DRAM
// )
void gemm_NEON_13x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 13] @DRAM
// )
void gemm_NEON_13x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 13] @DRAM
// )
void gemm_NEON_13x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 13] @DRAM
// )
void gemm_NEON_13x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 13] @DRAM
// )
void gemm_NEON_13x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 13] @DRAM
// )
void gemm_NEON_13x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 13] @DRAM
// )
void gemm_NEON_13x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 13] @DRAM
// )
void gemm_NEON_13x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 13] @DRAM
// )
void gemm_NEON_13x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 13] @DRAM
// )
void gemm_NEON_13x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 13] @DRAM
// )
void gemm_NEON_13x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 13] @DRAM
// )
void gemm_NEON_13x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 13] @DRAM
// )
void gemm_NEON_13x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 13] @DRAM
// )
void gemm_NEON_13x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 13] @DRAM
// )
void gemm_NEON_13x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_13x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 13] @DRAM
// )
void gemm_NEON_13x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 14] @DRAM
// )
void gemm_NEON_14x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 14] @DRAM
// )
void gemm_NEON_14x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 14] @DRAM
// )
void gemm_NEON_14x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 14] @DRAM
// )
void gemm_NEON_14x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 14] @DRAM
// )
void gemm_NEON_14x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 14] @DRAM
// )
void gemm_NEON_14x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 14] @DRAM
// )
void gemm_NEON_14x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 14] @DRAM
// )
void gemm_NEON_14x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 14] @DRAM
// )
void gemm_NEON_14x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 14] @DRAM
// )
void gemm_NEON_14x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 14] @DRAM
// )
void gemm_NEON_14x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 14] @DRAM
// )
void gemm_NEON_14x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 14] @DRAM
// )
void gemm_NEON_14x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 14] @DRAM
// )
void gemm_NEON_14x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 14] @DRAM
// )
void gemm_NEON_14x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 14] @DRAM
// )
void gemm_NEON_14x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 14] @DRAM
// )
void gemm_NEON_14x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 14] @DRAM
// )
void gemm_NEON_14x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 14] @DRAM
// )
void gemm_NEON_14x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 14] @DRAM
// )
void gemm_NEON_14x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 14] @DRAM
// )
void gemm_NEON_14x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 14] @DRAM
// )
void gemm_NEON_14x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 14] @DRAM
// )
void gemm_NEON_14x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 14] @DRAM
// )
void gemm_NEON_14x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 14] @DRAM
// )
void gemm_NEON_14x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 14] @DRAM
// )
void gemm_NEON_14x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 14] @DRAM
// )
void gemm_NEON_14x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 14] @DRAM
// )
void gemm_NEON_14x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 14] @DRAM
// )
void gemm_NEON_14x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 14] @DRAM
// )
void gemm_NEON_14x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 14] @DRAM
// )
void gemm_NEON_14x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 14] @DRAM
// )
void gemm_NEON_14x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 14] @DRAM
// )
void gemm_NEON_14x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 14] @DRAM
// )
void gemm_NEON_14x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 14] @DRAM
// )
void gemm_NEON_14x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 14] @DRAM
// )
void gemm_NEON_14x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 14] @DRAM
// )
void gemm_NEON_14x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 14] @DRAM
// )
void gemm_NEON_14x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 14] @DRAM
// )
void gemm_NEON_14x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 14] @DRAM
// )
void gemm_NEON_14x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 14] @DRAM
// )
void gemm_NEON_14x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 14] @DRAM
// )
void gemm_NEON_14x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 14] @DRAM
// )
void gemm_NEON_14x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 14] @DRAM
// )
void gemm_NEON_14x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 14] @DRAM
// )
void gemm_NEON_14x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 14] @DRAM
// )
void gemm_NEON_14x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 14] @DRAM
// )
void gemm_NEON_14x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_14x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 14] @DRAM
// )
void gemm_NEON_14x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 15] @DRAM
// )
void gemm_NEON_15x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 15] @DRAM
// )
void gemm_NEON_15x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 15] @DRAM
// )
void gemm_NEON_15x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 15] @DRAM
// )
void gemm_NEON_15x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 15] @DRAM
// )
void gemm_NEON_15x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 15] @DRAM
// )
void gemm_NEON_15x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 15] @DRAM
// )
void gemm_NEON_15x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 15] @DRAM
// )
void gemm_NEON_15x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 15] @DRAM
// )
void gemm_NEON_15x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 15] @DRAM
// )
void gemm_NEON_15x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 15] @DRAM
// )
void gemm_NEON_15x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 15] @DRAM
// )
void gemm_NEON_15x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 15] @DRAM
// )
void gemm_NEON_15x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 15] @DRAM
// )
void gemm_NEON_15x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 15] @DRAM
// )
void gemm_NEON_15x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 15] @DRAM
// )
void gemm_NEON_15x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 15] @DRAM
// )
void gemm_NEON_15x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 15] @DRAM
// )
void gemm_NEON_15x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 15] @DRAM
// )
void gemm_NEON_15x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 15] @DRAM
// )
void gemm_NEON_15x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 15] @DRAM
// )
void gemm_NEON_15x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 15] @DRAM
// )
void gemm_NEON_15x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 15] @DRAM
// )
void gemm_NEON_15x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 15] @DRAM
// )
void gemm_NEON_15x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 15] @DRAM
// )
void gemm_NEON_15x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 15] @DRAM
// )
void gemm_NEON_15x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 15] @DRAM
// )
void gemm_NEON_15x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 15] @DRAM
// )
void gemm_NEON_15x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 15] @DRAM
// )
void gemm_NEON_15x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 15] @DRAM
// )
void gemm_NEON_15x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 15] @DRAM
// )
void gemm_NEON_15x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 15] @DRAM
// )
void gemm_NEON_15x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 15] @DRAM
// )
void gemm_NEON_15x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 15] @DRAM
// )
void gemm_NEON_15x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 15] @DRAM
// )
void gemm_NEON_15x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 15] @DRAM
// )
void gemm_NEON_15x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 15] @DRAM
// )
void gemm_NEON_15x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 15] @DRAM
// )
void gemm_NEON_15x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 15] @DRAM
// )
void gemm_NEON_15x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 15] @DRAM
// )
void gemm_NEON_15x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 15] @DRAM
// )
void gemm_NEON_15x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 15] @DRAM
// )
void gemm_NEON_15x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 15] @DRAM
// )
void gemm_NEON_15x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 15] @DRAM
// )
void gemm_NEON_15x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 15] @DRAM
// )
void gemm_NEON_15x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 15] @DRAM
// )
void gemm_NEON_15x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 15] @DRAM
// )
void gemm_NEON_15x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_15x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 15] @DRAM
// )
void gemm_NEON_15x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 16] @DRAM
// )
void gemm_NEON_16x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 16] @DRAM
// )
void gemm_NEON_16x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 16] @DRAM
// )
void gemm_NEON_16x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 16] @DRAM
// )
void gemm_NEON_16x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 16] @DRAM
// )
void gemm_NEON_16x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 16] @DRAM
// )
void gemm_NEON_16x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 16] @DRAM
// )
void gemm_NEON_16x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 16] @DRAM
// )
void gemm_NEON_16x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 16] @DRAM
// )
void gemm_NEON_16x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 16] @DRAM
// )
void gemm_NEON_16x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 16] @DRAM
// )
void gemm_NEON_16x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 16] @DRAM
// )
void gemm_NEON_16x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 16] @DRAM
// )
void gemm_NEON_16x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 16] @DRAM
// )
void gemm_NEON_16x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 16] @DRAM
// )
void gemm_NEON_16x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 16] @DRAM
// )
void gemm_NEON_16x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 16] @DRAM
// )
void gemm_NEON_16x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 16] @DRAM
// )
void gemm_NEON_16x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 16] @DRAM
// )
void gemm_NEON_16x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 16] @DRAM
// )
void gemm_NEON_16x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 16] @DRAM
// )
void gemm_NEON_16x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 16] @DRAM
// )
void gemm_NEON_16x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 16] @DRAM
// )
void gemm_NEON_16x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 16] @DRAM
// )
void gemm_NEON_16x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 16] @DRAM
// )
void gemm_NEON_16x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 16] @DRAM
// )
void gemm_NEON_16x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 16] @DRAM
// )
void gemm_NEON_16x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 16] @DRAM
// )
void gemm_NEON_16x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 16] @DRAM
// )
void gemm_NEON_16x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 16] @DRAM
// )
void gemm_NEON_16x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 16] @DRAM
// )
void gemm_NEON_16x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 16] @DRAM
// )
void gemm_NEON_16x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 16] @DRAM
// )
void gemm_NEON_16x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 16] @DRAM
// )
void gemm_NEON_16x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 16] @DRAM
// )
void gemm_NEON_16x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 16] @DRAM
// )
void gemm_NEON_16x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 16] @DRAM
// )
void gemm_NEON_16x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 16] @DRAM
// )
void gemm_NEON_16x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 16] @DRAM
// )
void gemm_NEON_16x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 16] @DRAM
// )
void gemm_NEON_16x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 16] @DRAM
// )
void gemm_NEON_16x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 16] @DRAM
// )
void gemm_NEON_16x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 16] @DRAM
// )
void gemm_NEON_16x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 16] @DRAM
// )
void gemm_NEON_16x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 16] @DRAM
// )
void gemm_NEON_16x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 16] @DRAM
// )
void gemm_NEON_16x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_16x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 16] @DRAM
// )
void gemm_NEON_16x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_16x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 16] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 16] @DRAM
// )
void gemm_NEON_16x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );



#ifdef __cplusplus
}
#endif
#endif  // MIXED2_H
