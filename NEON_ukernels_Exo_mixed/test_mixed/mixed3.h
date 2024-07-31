
#pragma once
#ifndef MIXED3_H
#define MIXED3_H

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
// gemm_NEON_22x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 22] @DRAM
// )
void gemm_NEON_22x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 22] @DRAM
// )
void gemm_NEON_22x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 22] @DRAM
// )
void gemm_NEON_22x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 22] @DRAM
// )
void gemm_NEON_22x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 22] @DRAM
// )
void gemm_NEON_22x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 22] @DRAM
// )
void gemm_NEON_22x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 22] @DRAM
// )
void gemm_NEON_22x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 22] @DRAM
// )
void gemm_NEON_22x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 22] @DRAM
// )
void gemm_NEON_22x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 22] @DRAM
// )
void gemm_NEON_22x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 22] @DRAM
// )
void gemm_NEON_22x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 22] @DRAM
// )
void gemm_NEON_22x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 22] @DRAM
// )
void gemm_NEON_22x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 22] @DRAM
// )
void gemm_NEON_22x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 22] @DRAM
// )
void gemm_NEON_22x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 22] @DRAM
// )
void gemm_NEON_22x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 22] @DRAM
// )
void gemm_NEON_22x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 22] @DRAM
// )
void gemm_NEON_22x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 22] @DRAM
// )
void gemm_NEON_22x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 22] @DRAM
// )
void gemm_NEON_22x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 22] @DRAM
// )
void gemm_NEON_22x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 22] @DRAM
// )
void gemm_NEON_22x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 22] @DRAM
// )
void gemm_NEON_22x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 22] @DRAM
// )
void gemm_NEON_22x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 22] @DRAM
// )
void gemm_NEON_22x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 22] @DRAM
// )
void gemm_NEON_22x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 22] @DRAM
// )
void gemm_NEON_22x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 22] @DRAM
// )
void gemm_NEON_22x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 22] @DRAM
// )
void gemm_NEON_22x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 22] @DRAM
// )
void gemm_NEON_22x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 22] @DRAM
// )
void gemm_NEON_22x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 22] @DRAM
// )
void gemm_NEON_22x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 22] @DRAM
// )
void gemm_NEON_22x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 22] @DRAM
// )
void gemm_NEON_22x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 22] @DRAM
// )
void gemm_NEON_22x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 22] @DRAM
// )
void gemm_NEON_22x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 22] @DRAM
// )
void gemm_NEON_22x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 22] @DRAM
// )
void gemm_NEON_22x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 22] @DRAM
// )
void gemm_NEON_22x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 22] @DRAM
// )
void gemm_NEON_22x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 22] @DRAM
// )
void gemm_NEON_22x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 22] @DRAM
// )
void gemm_NEON_22x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 22] @DRAM
// )
void gemm_NEON_22x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 22] @DRAM
// )
void gemm_NEON_22x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 22] @DRAM
// )
void gemm_NEON_22x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 22] @DRAM
// )
void gemm_NEON_22x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 22] @DRAM
// )
void gemm_NEON_22x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_22x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 22] @DRAM
// )
void gemm_NEON_22x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 23] @DRAM
// )
void gemm_NEON_23x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 23] @DRAM
// )
void gemm_NEON_23x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 23] @DRAM
// )
void gemm_NEON_23x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 23] @DRAM
// )
void gemm_NEON_23x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 23] @DRAM
// )
void gemm_NEON_23x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 23] @DRAM
// )
void gemm_NEON_23x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 23] @DRAM
// )
void gemm_NEON_23x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 23] @DRAM
// )
void gemm_NEON_23x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 23] @DRAM
// )
void gemm_NEON_23x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 23] @DRAM
// )
void gemm_NEON_23x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 23] @DRAM
// )
void gemm_NEON_23x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 23] @DRAM
// )
void gemm_NEON_23x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 23] @DRAM
// )
void gemm_NEON_23x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][16, 23] @DRAM
// )
void gemm_NEON_23x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 23] @DRAM
// )
void gemm_NEON_23x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 23] @DRAM
// )
void gemm_NEON_23x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 23] @DRAM
// )
void gemm_NEON_23x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 23] @DRAM
// )
void gemm_NEON_23x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 23] @DRAM
// )
void gemm_NEON_23x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 23] @DRAM
// )
void gemm_NEON_23x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 23] @DRAM
// )
void gemm_NEON_23x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 23] @DRAM
// )
void gemm_NEON_23x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 23] @DRAM
// )
void gemm_NEON_23x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][20, 23] @DRAM
// )
void gemm_NEON_23x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 23] @DRAM
// )
void gemm_NEON_23x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 23] @DRAM
// )
void gemm_NEON_23x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 23] @DRAM
// )
void gemm_NEON_23x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 23] @DRAM
// )
void gemm_NEON_23x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 23] @DRAM
// )
void gemm_NEON_23x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 23] @DRAM
// )
void gemm_NEON_23x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 23] @DRAM
// )
void gemm_NEON_23x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][24, 23] @DRAM
// )
void gemm_NEON_23x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 23] @DRAM
// )
void gemm_NEON_23x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 23] @DRAM
// )
void gemm_NEON_23x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 23] @DRAM
// )
void gemm_NEON_23x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 23] @DRAM
// )
void gemm_NEON_23x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 23] @DRAM
// )
void gemm_NEON_23x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 23] @DRAM
// )
void gemm_NEON_23x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 23] @DRAM
// )
void gemm_NEON_23x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 23] @DRAM
// )
void gemm_NEON_23x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 23] @DRAM
// )
void gemm_NEON_23x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 23] @DRAM
// )
void gemm_NEON_23x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 23] @DRAM
// )
void gemm_NEON_23x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 23] @DRAM
// )
void gemm_NEON_23x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 23] @DRAM
// )
void gemm_NEON_23x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 23] @DRAM
// )
void gemm_NEON_23x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 23] @DRAM
// )
void gemm_NEON_23x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_23x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 23] @DRAM
// )
void gemm_NEON_23x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x10_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 24] @DRAM
// )
void gemm_NEON_24x10_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x10_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 24] @DRAM
// )
void gemm_NEON_24x10_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x11_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 24] @DRAM
// )
void gemm_NEON_24x11_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x11_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 24] @DRAM
// )
void gemm_NEON_24x11_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x12_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 24] @DRAM
// )
void gemm_NEON_24x12_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x12_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 24] @DRAM
// )
void gemm_NEON_24x12_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x13_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 24] @DRAM
// )
void gemm_NEON_24x13_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x13_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][13, 24] @DRAM
// )
void gemm_NEON_24x13_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x14_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 24] @DRAM
// )
void gemm_NEON_24x14_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x14_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][14, 24] @DRAM
// )
void gemm_NEON_24x14_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x15_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 24] @DRAM
// )
void gemm_NEON_24x15_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x15_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][15, 24] @DRAM
// )
void gemm_NEON_24x15_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x16_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 24] @DRAM
// )
void gemm_NEON_24x16_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x16_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 16] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][16, 24] @DRAM
// )
void gemm_NEON_24x16_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x17_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 24] @DRAM
// )
void gemm_NEON_24x17_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x17_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][17, 24] @DRAM
// )
void gemm_NEON_24x17_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x18_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 24] @DRAM
// )
void gemm_NEON_24x18_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x18_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][18, 24] @DRAM
// )
void gemm_NEON_24x18_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x19_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 24] @DRAM
// )
void gemm_NEON_24x19_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x19_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][19, 24] @DRAM
// )
void gemm_NEON_24x19_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x1_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 24] @DRAM
// )
void gemm_NEON_24x1_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x1_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 24] @DRAM
// )
void gemm_NEON_24x1_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x20_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 24] @DRAM
// )
void gemm_NEON_24x20_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x20_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 20] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][20, 24] @DRAM
// )
void gemm_NEON_24x20_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x21_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 24] @DRAM
// )
void gemm_NEON_24x21_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x21_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][21, 24] @DRAM
// )
void gemm_NEON_24x21_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x22_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 24] @DRAM
// )
void gemm_NEON_24x22_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x22_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][22, 24] @DRAM
// )
void gemm_NEON_24x22_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x23_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 24] @DRAM
// )
void gemm_NEON_24x23_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x23_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][23, 24] @DRAM
// )
void gemm_NEON_24x23_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x24_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 24] @DRAM
// )
void gemm_NEON_24x24_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x24_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 24] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][24, 24] @DRAM
// )
void gemm_NEON_24x24_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x2_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 24] @DRAM
// )
void gemm_NEON_24x2_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x2_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 24] @DRAM
// )
void gemm_NEON_24x2_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x3_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 24] @DRAM
// )
void gemm_NEON_24x3_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x3_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 24] @DRAM
// )
void gemm_NEON_24x3_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x4_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 24] @DRAM
// )
void gemm_NEON_24x4_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x4_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 24] @DRAM
// )
void gemm_NEON_24x4_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x5_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 24] @DRAM
// )
void gemm_NEON_24x5_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x5_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 24] @DRAM
// )
void gemm_NEON_24x5_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x6_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 24] @DRAM
// )
void gemm_NEON_24x6_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x6_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 24] @DRAM
// )
void gemm_NEON_24x6_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x7_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 24] @DRAM
// )
void gemm_NEON_24x7_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x7_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 24] @DRAM
// )
void gemm_NEON_24x7_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x8_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 24] @DRAM
// )
void gemm_NEON_24x8_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x8_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 24] @DRAM
// )
void gemm_NEON_24x8_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C );

// gemm_NEON_24x9_b0_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 24] @DRAM
// )
void gemm_NEON_24x9_b0_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );

// gemm_NEON_24x9_b1_col_i16(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 24] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 24] @DRAM
// )
void gemm_NEON_24x9_b1_col_i16( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci );



#ifdef __cplusplus
}
#endif
#endif  // MIXED3_H
