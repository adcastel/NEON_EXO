
#pragma once
#ifndef MIXED4_H
#define MIXED4_H

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
// gemm_NEON_17x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 17] @DRAM
// )
void gemm_NEON_17x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 17] @DRAM
// )
void gemm_NEON_17x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 17] @DRAM
// )
void gemm_NEON_17x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 17] @DRAM
// )
void gemm_NEON_17x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 17] @DRAM
// )
void gemm_NEON_17x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 17] @DRAM
// )
void gemm_NEON_17x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 17] @DRAM
// )
void gemm_NEON_17x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 17] @DRAM
// )
void gemm_NEON_17x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 17] @DRAM
// )
void gemm_NEON_17x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 17] @DRAM
// )
void gemm_NEON_17x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 17] @DRAM
// )
void gemm_NEON_17x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 17] @DRAM
// )
void gemm_NEON_17x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 17] @DRAM
// )
void gemm_NEON_17x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 17] @DRAM
// )
void gemm_NEON_17x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 17] @DRAM
// )
void gemm_NEON_17x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 17] @DRAM
// )
void gemm_NEON_17x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 17] @DRAM
// )
void gemm_NEON_17x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 17] @DRAM
// )
void gemm_NEON_17x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 17] @DRAM
// )
void gemm_NEON_17x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 17] @DRAM
// )
void gemm_NEON_17x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 17] @DRAM
// )
void gemm_NEON_17x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 17] @DRAM
// )
void gemm_NEON_17x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 17] @DRAM
// )
void gemm_NEON_17x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 17] @DRAM
// )
void gemm_NEON_17x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 17] @DRAM
// )
void gemm_NEON_17x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 17] @DRAM
// )
void gemm_NEON_17x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 17] @DRAM
// )
void gemm_NEON_17x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 17] @DRAM
// )
void gemm_NEON_17x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 17] @DRAM
// )
void gemm_NEON_17x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 17] @DRAM
// )
void gemm_NEON_17x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 17] @DRAM
// )
void gemm_NEON_17x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 17] @DRAM
// )
void gemm_NEON_17x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 17] @DRAM
// )
void gemm_NEON_17x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 17] @DRAM
// )
void gemm_NEON_17x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 17] @DRAM
// )
void gemm_NEON_17x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 17] @DRAM
// )
void gemm_NEON_17x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 17] @DRAM
// )
void gemm_NEON_17x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 17] @DRAM
// )
void gemm_NEON_17x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 17] @DRAM
// )
void gemm_NEON_17x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 17] @DRAM
// )
void gemm_NEON_17x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 17] @DRAM
// )
void gemm_NEON_17x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 17] @DRAM
// )
void gemm_NEON_17x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 17] @DRAM
// )
void gemm_NEON_17x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 17] @DRAM
// )
void gemm_NEON_17x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 17] @DRAM
// )
void gemm_NEON_17x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 17] @DRAM
// )
void gemm_NEON_17x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 17] @DRAM
// )
void gemm_NEON_17x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_17x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 17] @DRAM
// )
void gemm_NEON_17x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 18] @DRAM
// )
void gemm_NEON_18x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 18] @DRAM
// )
void gemm_NEON_18x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 18] @DRAM
// )
void gemm_NEON_18x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 18] @DRAM
// )
void gemm_NEON_18x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 18] @DRAM
// )
void gemm_NEON_18x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 18] @DRAM
// )
void gemm_NEON_18x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 18] @DRAM
// )
void gemm_NEON_18x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 18] @DRAM
// )
void gemm_NEON_18x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 18] @DRAM
// )
void gemm_NEON_18x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 18] @DRAM
// )
void gemm_NEON_18x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 18] @DRAM
// )
void gemm_NEON_18x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 18] @DRAM
// )
void gemm_NEON_18x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 18] @DRAM
// )
void gemm_NEON_18x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 18] @DRAM
// )
void gemm_NEON_18x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 18] @DRAM
// )
void gemm_NEON_18x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 18] @DRAM
// )
void gemm_NEON_18x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 18] @DRAM
// )
void gemm_NEON_18x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 18] @DRAM
// )
void gemm_NEON_18x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 18] @DRAM
// )
void gemm_NEON_18x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 18] @DRAM
// )
void gemm_NEON_18x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 18] @DRAM
// )
void gemm_NEON_18x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 18] @DRAM
// )
void gemm_NEON_18x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 18] @DRAM
// )
void gemm_NEON_18x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 18] @DRAM
// )
void gemm_NEON_18x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 18] @DRAM
// )
void gemm_NEON_18x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 18] @DRAM
// )
void gemm_NEON_18x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 18] @DRAM
// )
void gemm_NEON_18x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 18] @DRAM
// )
void gemm_NEON_18x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 18] @DRAM
// )
void gemm_NEON_18x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 18] @DRAM
// )
void gemm_NEON_18x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 18] @DRAM
// )
void gemm_NEON_18x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 18] @DRAM
// )
void gemm_NEON_18x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 18] @DRAM
// )
void gemm_NEON_18x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 18] @DRAM
// )
void gemm_NEON_18x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 18] @DRAM
// )
void gemm_NEON_18x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 18] @DRAM
// )
void gemm_NEON_18x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 18] @DRAM
// )
void gemm_NEON_18x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 18] @DRAM
// )
void gemm_NEON_18x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 18] @DRAM
// )
void gemm_NEON_18x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 18] @DRAM
// )
void gemm_NEON_18x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 18] @DRAM
// )
void gemm_NEON_18x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 18] @DRAM
// )
void gemm_NEON_18x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 18] @DRAM
// )
void gemm_NEON_18x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 18] @DRAM
// )
void gemm_NEON_18x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 18] @DRAM
// )
void gemm_NEON_18x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 18] @DRAM
// )
void gemm_NEON_18x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 18] @DRAM
// )
void gemm_NEON_18x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_18x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 18] @DRAM
// )
void gemm_NEON_18x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 19] @DRAM
// )
void gemm_NEON_19x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 19] @DRAM
// )
void gemm_NEON_19x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 19] @DRAM
// )
void gemm_NEON_19x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 19] @DRAM
// )
void gemm_NEON_19x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 19] @DRAM
// )
void gemm_NEON_19x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 19] @DRAM
// )
void gemm_NEON_19x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 19] @DRAM
// )
void gemm_NEON_19x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 19] @DRAM
// )
void gemm_NEON_19x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 19] @DRAM
// )
void gemm_NEON_19x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 19] @DRAM
// )
void gemm_NEON_19x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 19] @DRAM
// )
void gemm_NEON_19x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 19] @DRAM
// )
void gemm_NEON_19x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 19] @DRAM
// )
void gemm_NEON_19x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 19] @DRAM
// )
void gemm_NEON_19x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 19] @DRAM
// )
void gemm_NEON_19x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 19] @DRAM
// )
void gemm_NEON_19x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 19] @DRAM
// )
void gemm_NEON_19x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 19] @DRAM
// )
void gemm_NEON_19x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 19] @DRAM
// )
void gemm_NEON_19x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 19] @DRAM
// )
void gemm_NEON_19x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 19] @DRAM
// )
void gemm_NEON_19x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 19] @DRAM
// )
void gemm_NEON_19x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 19] @DRAM
// )
void gemm_NEON_19x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 19] @DRAM
// )
void gemm_NEON_19x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 19] @DRAM
// )
void gemm_NEON_19x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 19] @DRAM
// )
void gemm_NEON_19x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 19] @DRAM
// )
void gemm_NEON_19x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 19] @DRAM
// )
void gemm_NEON_19x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 19] @DRAM
// )
void gemm_NEON_19x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 19] @DRAM
// )
void gemm_NEON_19x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 19] @DRAM
// )
void gemm_NEON_19x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 19] @DRAM
// )
void gemm_NEON_19x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 19] @DRAM
// )
void gemm_NEON_19x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 19] @DRAM
// )
void gemm_NEON_19x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 19] @DRAM
// )
void gemm_NEON_19x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 19] @DRAM
// )
void gemm_NEON_19x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 19] @DRAM
// )
void gemm_NEON_19x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 19] @DRAM
// )
void gemm_NEON_19x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 19] @DRAM
// )
void gemm_NEON_19x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 19] @DRAM
// )
void gemm_NEON_19x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 19] @DRAM
// )
void gemm_NEON_19x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 19] @DRAM
// )
void gemm_NEON_19x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 19] @DRAM
// )
void gemm_NEON_19x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 19] @DRAM
// )
void gemm_NEON_19x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 19] @DRAM
// )
void gemm_NEON_19x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 19] @DRAM
// )
void gemm_NEON_19x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 19] @DRAM
// )
void gemm_NEON_19x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_19x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 19] @DRAM
// )
void gemm_NEON_19x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 20] @DRAM
// )
void gemm_NEON_20x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 20] @DRAM
// )
void gemm_NEON_20x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 20] @DRAM
// )
void gemm_NEON_20x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 20] @DRAM
// )
void gemm_NEON_20x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 20] @DRAM
// )
void gemm_NEON_20x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 20] @DRAM
// )
void gemm_NEON_20x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 20] @DRAM
// )
void gemm_NEON_20x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 20] @DRAM
// )
void gemm_NEON_20x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 20] @DRAM
// )
void gemm_NEON_20x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 20] @DRAM
// )
void gemm_NEON_20x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 20] @DRAM
// )
void gemm_NEON_20x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 20] @DRAM
// )
void gemm_NEON_20x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 20] @DRAM
// )
void gemm_NEON_20x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 20] @DRAM
// )
void gemm_NEON_20x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 20] @DRAM
// )
void gemm_NEON_20x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 20] @DRAM
// )
void gemm_NEON_20x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 20] @DRAM
// )
void gemm_NEON_20x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 20] @DRAM
// )
void gemm_NEON_20x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 20] @DRAM
// )
void gemm_NEON_20x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 20] @DRAM
// )
void gemm_NEON_20x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 20] @DRAM
// )
void gemm_NEON_20x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 20] @DRAM
// )
void gemm_NEON_20x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 20] @DRAM
// )
void gemm_NEON_20x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 20] @DRAM
// )
void gemm_NEON_20x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 20] @DRAM
// )
void gemm_NEON_20x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 20] @DRAM
// )
void gemm_NEON_20x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 20] @DRAM
// )
void gemm_NEON_20x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 20] @DRAM
// )
void gemm_NEON_20x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 20] @DRAM
// )
void gemm_NEON_20x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 20] @DRAM
// )
void gemm_NEON_20x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 20] @DRAM
// )
void gemm_NEON_20x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 20] @DRAM
// )
void gemm_NEON_20x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 20] @DRAM
// )
void gemm_NEON_20x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 20] @DRAM
// )
void gemm_NEON_20x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 20] @DRAM
// )
void gemm_NEON_20x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 20] @DRAM
// )
void gemm_NEON_20x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 20] @DRAM
// )
void gemm_NEON_20x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 20] @DRAM
// )
void gemm_NEON_20x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 20] @DRAM
// )
void gemm_NEON_20x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 20] @DRAM
// )
void gemm_NEON_20x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 20] @DRAM
// )
void gemm_NEON_20x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 20] @DRAM
// )
void gemm_NEON_20x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 20] @DRAM
// )
void gemm_NEON_20x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 20] @DRAM
// )
void gemm_NEON_20x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 20] @DRAM
// )
void gemm_NEON_20x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 20] @DRAM
// )
void gemm_NEON_20x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_20x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 20] @DRAM
// )
void gemm_NEON_20x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_20x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 20] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 20] @DRAM
// )
void gemm_NEON_20x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 21] @DRAM
// )
void gemm_NEON_21x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 21] @DRAM
// )
void gemm_NEON_21x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 21] @DRAM
// )
void gemm_NEON_21x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 21] @DRAM
// )
void gemm_NEON_21x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 21] @DRAM
// )
void gemm_NEON_21x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 21] @DRAM
// )
void gemm_NEON_21x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 21] @DRAM
// )
void gemm_NEON_21x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 21] @DRAM
// )
void gemm_NEON_21x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 21] @DRAM
// )
void gemm_NEON_21x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 21] @DRAM
// )
void gemm_NEON_21x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 21] @DRAM
// )
void gemm_NEON_21x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 21] @DRAM
// )
void gemm_NEON_21x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 21] @DRAM
// )
void gemm_NEON_21x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 21] @DRAM
// )
void gemm_NEON_21x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 21] @DRAM
// )
void gemm_NEON_21x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 21] @DRAM
// )
void gemm_NEON_21x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 21] @DRAM
// )
void gemm_NEON_21x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 21] @DRAM
// )
void gemm_NEON_21x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 21] @DRAM
// )
void gemm_NEON_21x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 21] @DRAM
// )
void gemm_NEON_21x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 21] @DRAM
// )
void gemm_NEON_21x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 21] @DRAM
// )
void gemm_NEON_21x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 21] @DRAM
// )
void gemm_NEON_21x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 21] @DRAM
// )
void gemm_NEON_21x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 21] @DRAM
// )
void gemm_NEON_21x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 21] @DRAM
// )
void gemm_NEON_21x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 21] @DRAM
// )
void gemm_NEON_21x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 21] @DRAM
// )
void gemm_NEON_21x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 21] @DRAM
// )
void gemm_NEON_21x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 21] @DRAM
// )
void gemm_NEON_21x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 21] @DRAM
// )
void gemm_NEON_21x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 21] @DRAM
// )
void gemm_NEON_21x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 21] @DRAM
// )
void gemm_NEON_21x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 21] @DRAM
// )
void gemm_NEON_21x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 21] @DRAM
// )
void gemm_NEON_21x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 21] @DRAM
// )
void gemm_NEON_21x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 21] @DRAM
// )
void gemm_NEON_21x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 21] @DRAM
// )
void gemm_NEON_21x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 21] @DRAM
// )
void gemm_NEON_21x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 21] @DRAM
// )
void gemm_NEON_21x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 21] @DRAM
// )
void gemm_NEON_21x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 21] @DRAM
// )
void gemm_NEON_21x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 21] @DRAM
// )
void gemm_NEON_21x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 21] @DRAM
// )
void gemm_NEON_21x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 21] @DRAM
// )
void gemm_NEON_21x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 21] @DRAM
// )
void gemm_NEON_21x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 21] @DRAM
// )
void gemm_NEON_21x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_21x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 21] @DRAM
// )
void gemm_NEON_21x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );



#ifdef __cplusplus
}
#endif
#endif  // MIXED4_H
