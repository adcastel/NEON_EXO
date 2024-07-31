#include "kernels_NEON_8x8_fp16.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


// gemm_NEON_1x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_1x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] = C[0];
free(C);
}

// gemm_NEON_1x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_1x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] += C[0];
free(C);
}

// gemm_NEON_1x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_1x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] = C[0];
Ci[ldci] = C[8];
free(C);
}

// gemm_NEON_1x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_1x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] += C[0];
Ci[ldci] += C[8];
free(C);
}

// gemm_NEON_1x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_1x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] = C[0];
Ci[ldci] = C[8];
Ci[2 * ldci] = C[16];
free(C);
}

// gemm_NEON_1x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_1x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] += C[0];
Ci[ldci] += C[8];
Ci[2 * ldci] += C[16];
free(C);
}

// gemm_NEON_1x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_1x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] = C[0];
Ci[ldci] = C[8];
Ci[2 * ldci] = C[16];
Ci[3 * ldci] = C[24];
free(C);
}

// gemm_NEON_1x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_1x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] += C[0];
Ci[ldci] += C[8];
Ci[2 * ldci] += C[16];
Ci[3 * ldci] += C[24];
free(C);
}

// gemm_NEON_1x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_1x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] = C[0];
Ci[ldci] = C[8];
Ci[2 * ldci] = C[16];
Ci[3 * ldci] = C[24];
Ci[4 * ldci] = C[32];
free(C);
}

// gemm_NEON_1x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_1x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] += C[0];
Ci[ldci] += C[8];
Ci[2 * ldci] += C[16];
Ci[3 * ldci] += C[24];
Ci[4 * ldci] += C[32];
free(C);
}

// gemm_NEON_1x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_1x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] = C[0];
Ci[ldci] = C[8];
Ci[2 * ldci] = C[16];
Ci[3 * ldci] = C[24];
Ci[4 * ldci] = C[32];
Ci[5 * ldci] = C[40];
free(C);
}

// gemm_NEON_1x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_1x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] += C[0];
Ci[ldci] += C[8];
Ci[2 * ldci] += C[16];
Ci[3 * ldci] += C[24];
Ci[4 * ldci] += C[32];
Ci[5 * ldci] += C[40];
free(C);
}

// gemm_NEON_1x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_1x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] = C[0];
Ci[ldci] = C[8];
Ci[2 * ldci] = C[16];
Ci[3 * ldci] = C[24];
Ci[4 * ldci] = C[32];
Ci[5 * ldci] = C[40];
Ci[6 * ldci] = C[48];
free(C);
}

// gemm_NEON_1x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_1x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] += C[0];
Ci[ldci] += C[8];
Ci[2 * ldci] += C[16];
Ci[3 * ldci] += C[24];
Ci[4 * ldci] += C[32];
Ci[5 * ldci] += C[40];
Ci[6 * ldci] += C[48];
free(C);
}

// gemm_NEON_1x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_1x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] = C[0];
Ci[ldci] = C[8];
Ci[2 * ldci] = C[16];
Ci[3 * ldci] = C[24];
Ci[4 * ldci] = C[32];
Ci[5 * ldci] = C[40];
Ci[6 * ldci] = C[48];
Ci[7 * ldci] = C[56];
free(C);
}

// gemm_NEON_1x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_1x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] += C[0];
Ci[ldci] += C[8];
Ci[2 * ldci] += C[16];
Ci[3 * ldci] += C[24];
Ci[4 * ldci] += C[32];
Ci[5 * ldci] += C[40];
Ci[6 * ldci] += C[48];
Ci[7 * ldci] += C[56];
free(C);
}

// gemm_NEON_2x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_2x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
free(C);
}

// gemm_NEON_2x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_2x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
free(C);
}

// gemm_NEON_2x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_2x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
free(C);
}

// gemm_NEON_2x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_2x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
free(C);
}

// gemm_NEON_2x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_2x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
free(C);
}

// gemm_NEON_2x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_2x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
free(C);
}

// gemm_NEON_2x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_2x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
free(C);
}

// gemm_NEON_2x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_2x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
free(C);
}

// gemm_NEON_2x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_2x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
free(C);
}

// gemm_NEON_2x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_2x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
free(C);
}

// gemm_NEON_2x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_2x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
free(C);
}

// gemm_NEON_2x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_2x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
free(C);
}

// gemm_NEON_2x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_2x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
free(C);
}

// gemm_NEON_2x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_2x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
free(C);
}

// gemm_NEON_2x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_2x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
free(C);
}

// gemm_NEON_2x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_2x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
free(C);
}

// gemm_NEON_3x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_3x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
free(C);
}

// gemm_NEON_3x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_3x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
free(C);
}

// gemm_NEON_3x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_3x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
free(C);
}

// gemm_NEON_3x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_3x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
free(C);
}

// gemm_NEON_3x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_3x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
free(C);
}

// gemm_NEON_3x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_3x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
free(C);
}

// gemm_NEON_3x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_3x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
free(C);
}

// gemm_NEON_3x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_3x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
free(C);
}

// gemm_NEON_3x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_3x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
free(C);
}

// gemm_NEON_3x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_3x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
free(C);
}

// gemm_NEON_3x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_3x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
free(C);
}

// gemm_NEON_3x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_3x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
free(C);
}

// gemm_NEON_3x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_3x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
free(C);
}

// gemm_NEON_3x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_3x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
free(C);
}

// gemm_NEON_3x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_3x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
Ci[7 * ldci + 2] = C[58];
free(C);
}

// gemm_NEON_3x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_3x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
Ci[7 * ldci + 2] += C[58];
free(C);
}

// gemm_NEON_4x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_4x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
free(C);
}

// gemm_NEON_4x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_4x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
free(C);
}

// gemm_NEON_4x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_4x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
free(C);
}

// gemm_NEON_4x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_4x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
free(C);
}

// gemm_NEON_4x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_4x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
free(C);
}

// gemm_NEON_4x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_4x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
free(C);
}

// gemm_NEON_4x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_4x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
free(C);
}

// gemm_NEON_4x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_4x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
free(C);
}

// gemm_NEON_4x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_4x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
free(C);
}

// gemm_NEON_4x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_4x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
free(C);
}

// gemm_NEON_4x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_4x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
free(C);
}

// gemm_NEON_4x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_4x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
free(C);
}

// gemm_NEON_4x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_4x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
free(C);
}

// gemm_NEON_4x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_4x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
free(C);
}

// gemm_NEON_4x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_4x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
Ci[7 * ldci + 2] = C[58];
Ci[7 * ldci + 3] = C[59];
free(C);
}

// gemm_NEON_4x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_4x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
Ci[7 * ldci + 2] += C[58];
Ci[7 * ldci + 3] += C[59];
free(C);
}

// gemm_NEON_5x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_5x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
free(C);
}

// gemm_NEON_5x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_5x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
free(C);
}

// gemm_NEON_5x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_5x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
free(C);
}

// gemm_NEON_5x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_5x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
free(C);
}

// gemm_NEON_5x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_5x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
free(C);
}

// gemm_NEON_5x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_5x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
free(C);
}

// gemm_NEON_5x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_5x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
free(C);
}

// gemm_NEON_5x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_5x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
free(C);
}

// gemm_NEON_5x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_5x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
free(C);
}

// gemm_NEON_5x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_5x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
free(C);
}

// gemm_NEON_5x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_5x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
free(C);
}

// gemm_NEON_5x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_5x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
free(C);
}

// gemm_NEON_5x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_5x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
Ci[6 * ldci + 4] = C[52];
free(C);
}

// gemm_NEON_5x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_5x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
Ci[6 * ldci + 4] += C[52];
free(C);
}

// gemm_NEON_5x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_5x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
Ci[6 * ldci + 4] = C[52];
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
Ci[7 * ldci + 2] = C[58];
Ci[7 * ldci + 3] = C[59];
Ci[7 * ldci + 4] = C[60];
free(C);
}

// gemm_NEON_5x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_5x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
Ci[6 * ldci + 4] += C[52];
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
Ci[7 * ldci + 2] += C[58];
Ci[7 * ldci + 3] += C[59];
Ci[7 * ldci + 4] += C[60];
free(C);
}

// gemm_NEON_6x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_6x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
free(C);
}

// gemm_NEON_6x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_6x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
free(C);
}

// gemm_NEON_6x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_6x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
free(C);
}

// gemm_NEON_6x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_6x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
free(C);
}

// gemm_NEON_6x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_6x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
free(C);
}

// gemm_NEON_6x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_6x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
free(C);
}

// gemm_NEON_6x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_6x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
free(C);
}

// gemm_NEON_6x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_6x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
free(C);
}

// gemm_NEON_6x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_6x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
free(C);
}

// gemm_NEON_6x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_6x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
free(C);
}

// gemm_NEON_6x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_6x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[5 * ldci + 5] = C[45];
free(C);
}

// gemm_NEON_6x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_6x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[5 * ldci + 5] += C[45];
free(C);
}

// gemm_NEON_6x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_6x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[5 * ldci + 5] = C[45];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
Ci[6 * ldci + 4] = C[52];
Ci[6 * ldci + 5] = C[53];
free(C);
}

// gemm_NEON_6x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_6x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[5 * ldci + 5] += C[45];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
Ci[6 * ldci + 4] += C[52];
Ci[6 * ldci + 5] += C[53];
free(C);
}

// gemm_NEON_6x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_6x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[5 * ldci + 5] = C[45];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
Ci[6 * ldci + 4] = C[52];
Ci[6 * ldci + 5] = C[53];
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
Ci[7 * ldci + 2] = C[58];
Ci[7 * ldci + 3] = C[59];
Ci[7 * ldci + 4] = C[60];
Ci[7 * ldci + 5] = C[61];
free(C);
}

// gemm_NEON_6x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_6x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[5 * ldci + 5] += C[45];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
Ci[6 * ldci + 4] += C[52];
Ci[6 * ldci + 5] += C[53];
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
Ci[7 * ldci + 2] += C[58];
Ci[7 * ldci + 3] += C[59];
Ci[7 * ldci + 4] += C[60];
Ci[7 * ldci + 5] += C[61];
free(C);
}

// gemm_NEON_7x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_7x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
free(C);
}

// gemm_NEON_7x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_7x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
free(C);
}

// gemm_NEON_7x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_7x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
free(C);
}

// gemm_NEON_7x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_7x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
free(C);
}

// gemm_NEON_7x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_7x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
free(C);
}

// gemm_NEON_7x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_7x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
free(C);
}

// gemm_NEON_7x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_7x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
free(C);
}

// gemm_NEON_7x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_7x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
free(C);
}

// gemm_NEON_7x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_7x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[4 * ldci + 6] = C[38];
free(C);
}

// gemm_NEON_7x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_7x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[4 * ldci + 6] += C[38];
free(C);
}

// gemm_NEON_7x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_7x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[4 * ldci + 6] = C[38];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[5 * ldci + 5] = C[45];
Ci[5 * ldci + 6] = C[46];
free(C);
}

// gemm_NEON_7x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_7x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[4 * ldci + 6] += C[38];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[5 * ldci + 5] += C[45];
Ci[5 * ldci + 6] += C[46];
free(C);
}

// gemm_NEON_7x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_7x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[4 * ldci + 6] = C[38];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[5 * ldci + 5] = C[45];
Ci[5 * ldci + 6] = C[46];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
Ci[6 * ldci + 4] = C[52];
Ci[6 * ldci + 5] = C[53];
Ci[6 * ldci + 6] = C[54];
free(C);
}

// gemm_NEON_7x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_7x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[4 * ldci + 6] += C[38];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[5 * ldci + 5] += C[45];
Ci[5 * ldci + 6] += C[46];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
Ci[6 * ldci + 4] += C[52];
Ci[6 * ldci + 5] += C[53];
Ci[6 * ldci + 6] += C[54];
free(C);
}

// gemm_NEON_7x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_7x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[4 * ldci + 6] = C[38];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[5 * ldci + 5] = C[45];
Ci[5 * ldci + 6] = C[46];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
Ci[6 * ldci + 4] = C[52];
Ci[6 * ldci + 5] = C[53];
Ci[6 * ldci + 6] = C[54];
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
Ci[7 * ldci + 2] = C[58];
Ci[7 * ldci + 3] = C[59];
Ci[7 * ldci + 4] = C[60];
Ci[7 * ldci + 5] = C[61];
Ci[7 * ldci + 6] = C[62];
free(C);
}

// gemm_NEON_7x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][8, 8] @DRAM
// )
void gemm_NEON_7x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
vst1q_f16(&C[(7) * 8], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[4 * ldci + 6] += C[38];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[5 * ldci + 5] += C[45];
Ci[5 * ldci + 6] += C[46];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
Ci[6 * ldci + 4] += C[52];
Ci[6 * ldci + 5] += C[53];
Ci[6 * ldci + 6] += C[54];
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
Ci[7 * ldci + 2] += C[58];
Ci[7 * ldci + 3] += C[59];
Ci[7 * ldci + 4] += C[60];
Ci[7 * ldci + 5] += C[61];
Ci[7 * ldci + 6] += C[62];
free(C);
}

// gemm_NEON_8x1_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_8x1_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
free(C);
}

// gemm_NEON_8x1_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][1, 8] @DRAM
// )
void gemm_NEON_8x1_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f16(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
free(C);
}

// gemm_NEON_8x2_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_8x2_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[ldci + 7] = C[15];
free(C);
}

// gemm_NEON_8x2_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][2, 8] @DRAM
// )
void gemm_NEON_8x2_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[ldci + 7] += C[15];
free(C);
}

// gemm_NEON_8x3_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_8x3_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[ldci + 7] = C[15];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[2 * ldci + 7] = C[23];
free(C);
}

// gemm_NEON_8x3_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][3, 8] @DRAM
// )
void gemm_NEON_8x3_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[ldci + 7] += C[15];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[2 * ldci + 7] += C[23];
free(C);
}

// gemm_NEON_8x4_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_8x4_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[ldci + 7] = C[15];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[2 * ldci + 7] = C[23];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
Ci[3 * ldci + 7] = C[31];
free(C);
}

// gemm_NEON_8x4_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][4, 8] @DRAM
// )
void gemm_NEON_8x4_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[ldci + 7] += C[15];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[2 * ldci + 7] += C[23];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
Ci[3 * ldci + 7] += C[31];
free(C);
}

// gemm_NEON_8x5_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_8x5_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[ldci + 7] = C[15];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[2 * ldci + 7] = C[23];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
Ci[3 * ldci + 7] = C[31];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[4 * ldci + 6] = C[38];
Ci[4 * ldci + 7] = C[39];
free(C);
}

// gemm_NEON_8x5_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 5] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][5, 8] @DRAM
// )
void gemm_NEON_8x5_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[ldci + 7] += C[15];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[2 * ldci + 7] += C[23];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
Ci[3 * ldci + 7] += C[31];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[4 * ldci + 6] += C[38];
Ci[4 * ldci + 7] += C[39];
free(C);
}

// gemm_NEON_8x6_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_8x6_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[ldci + 7] = C[15];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[2 * ldci + 7] = C[23];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
Ci[3 * ldci + 7] = C[31];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[4 * ldci + 6] = C[38];
Ci[4 * ldci + 7] = C[39];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[5 * ldci + 5] = C[45];
Ci[5 * ldci + 6] = C[46];
Ci[5 * ldci + 7] = C[47];
free(C);
}

// gemm_NEON_8x6_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 6] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][6, 8] @DRAM
// )
void gemm_NEON_8x6_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[ldci + 7] += C[15];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[2 * ldci + 7] += C[23];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
Ci[3 * ldci + 7] += C[31];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[4 * ldci + 6] += C[38];
Ci[4 * ldci + 7] += C[39];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[5 * ldci + 5] += C[45];
Ci[5 * ldci + 6] += C[46];
Ci[5 * ldci + 7] += C[47];
free(C);
}

// gemm_NEON_8x7_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_8x7_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[ldci] = C[8];
Ci[ldci + 1] = C[9];
Ci[ldci + 2] = C[10];
Ci[ldci + 3] = C[11];
Ci[ldci + 4] = C[12];
Ci[ldci + 5] = C[13];
Ci[ldci + 6] = C[14];
Ci[ldci + 7] = C[15];
Ci[2 * ldci] = C[16];
Ci[2 * ldci + 1] = C[17];
Ci[2 * ldci + 2] = C[18];
Ci[2 * ldci + 3] = C[19];
Ci[2 * ldci + 4] = C[20];
Ci[2 * ldci + 5] = C[21];
Ci[2 * ldci + 6] = C[22];
Ci[2 * ldci + 7] = C[23];
Ci[3 * ldci] = C[24];
Ci[3 * ldci + 1] = C[25];
Ci[3 * ldci + 2] = C[26];
Ci[3 * ldci + 3] = C[27];
Ci[3 * ldci + 4] = C[28];
Ci[3 * ldci + 5] = C[29];
Ci[3 * ldci + 6] = C[30];
Ci[3 * ldci + 7] = C[31];
Ci[4 * ldci] = C[32];
Ci[4 * ldci + 1] = C[33];
Ci[4 * ldci + 2] = C[34];
Ci[4 * ldci + 3] = C[35];
Ci[4 * ldci + 4] = C[36];
Ci[4 * ldci + 5] = C[37];
Ci[4 * ldci + 6] = C[38];
Ci[4 * ldci + 7] = C[39];
Ci[5 * ldci] = C[40];
Ci[5 * ldci + 1] = C[41];
Ci[5 * ldci + 2] = C[42];
Ci[5 * ldci + 3] = C[43];
Ci[5 * ldci + 4] = C[44];
Ci[5 * ldci + 5] = C[45];
Ci[5 * ldci + 6] = C[46];
Ci[5 * ldci + 7] = C[47];
Ci[6 * ldci] = C[48];
Ci[6 * ldci + 1] = C[49];
Ci[6 * ldci + 2] = C[50];
Ci[6 * ldci + 3] = C[51];
Ci[6 * ldci + 4] = C[52];
Ci[6 * ldci + 5] = C[53];
Ci[6 * ldci + 6] = C[54];
Ci[6 * ldci + 7] = C[55];
free(C);
}

// gemm_NEON_8x7_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 7] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f16][7, 8] @DRAM
// )
void gemm_NEON_8x7_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* b, _Float16 * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
_Float16 *C = (_Float16*) malloc(8 * 8 * sizeof(*C));
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[8], C_reg_1);
vst1q_f16(&C[(2) * 8], C_reg_2);
vst1q_f16(&C[(3) * 8], C_reg_3);
vst1q_f16(&C[(4) * 8], C_reg_4);
vst1q_f16(&C[(5) * 8], C_reg_5);
vst1q_f16(&C[(6) * 8], C_reg_6);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[ldci] += C[8];
Ci[ldci + 1] += C[9];
Ci[ldci + 2] += C[10];
Ci[ldci + 3] += C[11];
Ci[ldci + 4] += C[12];
Ci[ldci + 5] += C[13];
Ci[ldci + 6] += C[14];
Ci[ldci + 7] += C[15];
Ci[2 * ldci] += C[16];
Ci[2 * ldci + 1] += C[17];
Ci[2 * ldci + 2] += C[18];
Ci[2 * ldci + 3] += C[19];
Ci[2 * ldci + 4] += C[20];
Ci[2 * ldci + 5] += C[21];
Ci[2 * ldci + 6] += C[22];
Ci[2 * ldci + 7] += C[23];
Ci[3 * ldci] += C[24];
Ci[3 * ldci + 1] += C[25];
Ci[3 * ldci + 2] += C[26];
Ci[3 * ldci + 3] += C[27];
Ci[3 * ldci + 4] += C[28];
Ci[3 * ldci + 5] += C[29];
Ci[3 * ldci + 6] += C[30];
Ci[3 * ldci + 7] += C[31];
Ci[4 * ldci] += C[32];
Ci[4 * ldci + 1] += C[33];
Ci[4 * ldci + 2] += C[34];
Ci[4 * ldci + 3] += C[35];
Ci[4 * ldci + 4] += C[36];
Ci[4 * ldci + 5] += C[37];
Ci[4 * ldci + 6] += C[38];
Ci[4 * ldci + 7] += C[39];
Ci[5 * ldci] += C[40];
Ci[5 * ldci + 1] += C[41];
Ci[5 * ldci + 2] += C[42];
Ci[5 * ldci + 3] += C[43];
Ci[5 * ldci + 4] += C[44];
Ci[5 * ldci + 5] += C[45];
Ci[5 * ldci + 6] += C[46];
Ci[5 * ldci + 7] += C[47];
Ci[6 * ldci] += C[48];
Ci[6 * ldci + 1] += C[49];
Ci[6 * ldci + 2] += C[50];
Ci[6 * ldci + 3] += C[51];
Ci[6 * ldci + 4] += C[52];
Ci[6 * ldci + 5] += C[53];
Ci[6 * ldci + 6] += C[54];
Ci[6 * ldci + 7] += C[55];
free(C);
}

// gemm_NEON_8x8_b0_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_NEON_8x8_b0_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vmovq_n_f16((_Float16)0.0);
C_reg_1 = vmovq_n_f16((_Float16)0.0);
C_reg_2 = vmovq_n_f16((_Float16)0.0);
C_reg_3 = vmovq_n_f16((_Float16)0.0);
C_reg_4 = vmovq_n_f16((_Float16)0.0);
C_reg_5 = vmovq_n_f16((_Float16)0.0);
C_reg_6 = vmovq_n_f16((_Float16)0.0);
C_reg_7 = vmovq_n_f16((_Float16)0.0);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[ldc], C_reg_1);
vst1q_f16(&C[(2) * (ldc)], C_reg_2);
vst1q_f16(&C[(3) * (ldc)], C_reg_3);
vst1q_f16(&C[(4) * (ldc)], C_reg_4);
vst1q_f16(&C[(5) * (ldc)], C_reg_5);
vst1q_f16(&C[(6) * (ldc)], C_reg_6);
vst1q_f16(&C[(7) * (ldc)], C_reg_7);
}

// gemm_NEON_8x8_b1_col_fp16(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f16][KC, 8] @DRAM,
//     B : [f16][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f16][8, 8] @DRAM
// )
void gemm_NEON_8x8_b1_col_fp16( void *ctxt, int_fast32_t KC, const float* alpha, _Float16 * A, int lda, _Float16 * B, int ldb, const float* beta, _Float16 C, int ldc ) {
float16x8_t C_reg_0;
float16x8_t C_reg_1;
float16x8_t C_reg_2;
float16x8_t C_reg_3;
float16x8_t C_reg_4;
float16x8_t C_reg_5;
float16x8_t C_reg_6;
float16x8_t C_reg_7;
C_reg_0 = vld1q_f16(&C[0]);
C_reg_1 = vld1q_f16(&C[ldc]);
C_reg_2 = vld1q_f16(&C[(2) * (ldc)]);
C_reg_3 = vld1q_f16(&C[(3) * (ldc)]);
C_reg_4 = vld1q_f16(&C[(4) * (ldc)]);
C_reg_5 = vld1q_f16(&C[(5) * (ldc)]);
C_reg_6 = vld1q_f16(&C[(6) * (ldc)]);
C_reg_7 = vld1q_f16(&C[(7) * (ldc)]);
float16x8_t A_reg;
float16x8_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f16(&A[(k) * (8)]);
  B_reg = vld1q_f16(&B[(k) * (8)]);
  C_reg_0 = vfmaq_laneq_f16(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f16(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f16(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f16(C_reg_3, A_reg, B_reg, (3));
  C_reg_4 = vfmaq_laneq_f16(C_reg_4, A_reg, B_reg, (4));
  C_reg_5 = vfmaq_laneq_f16(C_reg_5, A_reg, B_reg, (5));
  C_reg_6 = vfmaq_laneq_f16(C_reg_6, A_reg, B_reg, (6));
  C_reg_7 = vfmaq_laneq_f16(C_reg_7, A_reg, B_reg, (7));
}
vst1q_f16(&C[0], C_reg_0);
vst1q_f16(&C[ldc], C_reg_1);
vst1q_f16(&C[(2) * (ldc)], C_reg_2);
vst1q_f16(&C[(3) * (ldc)], C_reg_3);
vst1q_f16(&C[(4) * (ldc)], C_reg_4);
vst1q_f16(&C[(5) * (ldc)], C_reg_5);
vst1q_f16(&C[(6) * (ldc)], C_reg_6);
vst1q_f16(&C[(7) * (ldc)], C_reg_7);
}


/* relying on the following instruction..."
neon_vfmla_8xf16_8xf16_ori(dst,lhs,rhs,jtt)
{dst_data} = vfmaq_laneq_f16({dst_data}, {lhs_data}, {rhs_data}, {jtt});
*/

/* relying on the following instruction..."
neon_vld_8xf16(dst,src,e)
{dst_data} = vld1q_f16(&{src_data});
*/

/* relying on the following instruction..."
neon_vst_8xf16(dst,src,e)
vst1q_f16(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
neon_zero_8xf16_new(dst)
{dst_data} = vmovq_n_f16((_Float16)0.0);
*/
