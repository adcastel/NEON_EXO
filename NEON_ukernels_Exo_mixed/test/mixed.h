
#pragma once
#ifndef MIXED_H
#define MIXED_H

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



#ifdef __cplusplus
}
#endif
#endif  // MIXED_H
