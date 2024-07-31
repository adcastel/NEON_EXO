#include "kernels_NEON_8x20_fp32.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


// gemm_NEON_1x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
free(C);
}

// gemm_NEON_1x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
free(C);
}

// gemm_NEON_1x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
free(C);
}

// gemm_NEON_1x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
free(C);
}

// gemm_NEON_1x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
free(C);
}

// gemm_NEON_1x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
free(C);
}

// gemm_NEON_1x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 1] @DRAM
// )
void gemm_NEON_1x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
Ci[12 * ldci] = C[48];
free(C);
}

// gemm_NEON_1x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 1] @DRAM
// )
void gemm_NEON_1x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
Ci[12 * ldci] += C[48];
free(C);
}

// gemm_NEON_1x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 1] @DRAM
// )
void gemm_NEON_1x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
Ci[12 * ldci] = C[48];
Ci[13 * ldci] = C[52];
free(C);
}

// gemm_NEON_1x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 1] @DRAM
// )
void gemm_NEON_1x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
Ci[12 * ldci] += C[48];
Ci[13 * ldci] += C[52];
free(C);
}

// gemm_NEON_1x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 1] @DRAM
// )
void gemm_NEON_1x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
Ci[12 * ldci] = C[48];
Ci[13 * ldci] = C[52];
Ci[14 * ldci] = C[56];
free(C);
}

// gemm_NEON_1x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 1] @DRAM
// )
void gemm_NEON_1x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
Ci[12 * ldci] += C[48];
Ci[13 * ldci] += C[52];
Ci[14 * ldci] += C[56];
free(C);
}

// gemm_NEON_1x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 1] @DRAM
// )
void gemm_NEON_1x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
Ci[12 * ldci] = C[48];
Ci[13 * ldci] = C[52];
Ci[14 * ldci] = C[56];
Ci[15 * ldci] = C[60];
free(C);
}

// gemm_NEON_1x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 1] @DRAM
// )
void gemm_NEON_1x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
Ci[12 * ldci] += C[48];
Ci[13 * ldci] += C[52];
Ci[14 * ldci] += C[56];
Ci[15 * ldci] += C[60];
free(C);
}

// gemm_NEON_1x17_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 1] @DRAM
// )
void gemm_NEON_1x17_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
Ci[12 * ldci] = C[48];
Ci[13 * ldci] = C[52];
Ci[14 * ldci] = C[56];
Ci[15 * ldci] = C[60];
Ci[16 * ldci] = C[64];
free(C);
}

// gemm_NEON_1x17_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 1] @DRAM
// )
void gemm_NEON_1x17_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
Ci[12 * ldci] += C[48];
Ci[13 * ldci] += C[52];
Ci[14 * ldci] += C[56];
Ci[15 * ldci] += C[60];
Ci[16 * ldci] += C[64];
free(C);
}

// gemm_NEON_1x18_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 1] @DRAM
// )
void gemm_NEON_1x18_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
Ci[12 * ldci] = C[48];
Ci[13 * ldci] = C[52];
Ci[14 * ldci] = C[56];
Ci[15 * ldci] = C[60];
Ci[16 * ldci] = C[64];
Ci[17 * ldci] = C[68];
free(C);
}

// gemm_NEON_1x18_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 1] @DRAM
// )
void gemm_NEON_1x18_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
Ci[12 * ldci] += C[48];
Ci[13 * ldci] += C[52];
Ci[14 * ldci] += C[56];
Ci[15 * ldci] += C[60];
Ci[16 * ldci] += C[64];
Ci[17 * ldci] += C[68];
free(C);
}

// gemm_NEON_1x19_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 1] @DRAM
// )
void gemm_NEON_1x19_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
Ci[12 * ldci] = C[48];
Ci[13 * ldci] = C[52];
Ci[14 * ldci] = C[56];
Ci[15 * ldci] = C[60];
Ci[16 * ldci] = C[64];
Ci[17 * ldci] = C[68];
Ci[18 * ldci] = C[72];
free(C);
}

// gemm_NEON_1x19_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 1] @DRAM
// )
void gemm_NEON_1x19_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
Ci[12 * ldci] += C[48];
Ci[13 * ldci] += C[52];
Ci[14 * ldci] += C[56];
Ci[15 * ldci] += C[60];
Ci[16 * ldci] += C[64];
Ci[17 * ldci] += C[68];
Ci[18 * ldci] += C[72];
free(C);
}

// gemm_NEON_1x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_1x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
C_reg_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] = C[0];
free(C);
}

// gemm_NEON_1x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_1x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
C_reg_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] += C[0];
free(C);
}

// gemm_NEON_1x20_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 1] @DRAM
// )
void gemm_NEON_1x20_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
Ci[9 * ldci] = C[36];
Ci[10 * ldci] = C[40];
Ci[11 * ldci] = C[44];
Ci[12 * ldci] = C[48];
Ci[13 * ldci] = C[52];
Ci[14 * ldci] = C[56];
Ci[15 * ldci] = C[60];
Ci[16 * ldci] = C[64];
Ci[17 * ldci] = C[68];
Ci[18 * ldci] = C[72];
Ci[19 * ldci] = C[76];
free(C);
}

// gemm_NEON_1x20_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 1] @DRAM
// )
void gemm_NEON_1x20_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
Ci[9 * ldci] += C[36];
Ci[10 * ldci] += C[40];
Ci[11 * ldci] += C[44];
Ci[12 * ldci] += C[48];
Ci[13 * ldci] += C[52];
Ci[14 * ldci] += C[56];
Ci[15 * ldci] += C[60];
Ci[16 * ldci] += C[64];
Ci[17 * ldci] += C[68];
Ci[18 * ldci] += C[72];
Ci[19 * ldci] += C[76];
free(C);
}

// gemm_NEON_1x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_1x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
Ci[0] = C[0];
Ci[ldci] = C[4];
free(C);
}

// gemm_NEON_1x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_1x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
Ci[0] += C[0];
Ci[ldci] += C[4];
free(C);
}

// gemm_NEON_1x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_1x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
free(C);
}

// gemm_NEON_1x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_1x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
free(C);
}

// gemm_NEON_1x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_1x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
free(C);
}

// gemm_NEON_1x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_1x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
free(C);
}

// gemm_NEON_1x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
free(C);
}

// gemm_NEON_1x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
free(C);
}

// gemm_NEON_1x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
free(C);
}

// gemm_NEON_1x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
free(C);
}

// gemm_NEON_1x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
free(C);
}

// gemm_NEON_1x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
free(C);
}

// gemm_NEON_1x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
free(C);
}

// gemm_NEON_1x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
free(C);
}

// gemm_NEON_1x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[ldci] = C[4];
Ci[2 * ldci] = C[8];
Ci[3 * ldci] = C[12];
Ci[4 * ldci] = C[16];
Ci[5 * ldci] = C[20];
Ci[6 * ldci] = C[24];
Ci[7 * ldci] = C[28];
Ci[8 * ldci] = C[32];
free(C);
}

// gemm_NEON_1x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[ldci] += C[4];
Ci[2 * ldci] += C[8];
Ci[3 * ldci] += C[12];
Ci[4 * ldci] += C[16];
Ci[5 * ldci] += C[20];
Ci[6 * ldci] += C[24];
Ci[7 * ldci] += C[28];
Ci[8 * ldci] += C[32];
free(C);
}

// gemm_NEON_2x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
free(C);
}

// gemm_NEON_2x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
free(C);
}

// gemm_NEON_2x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
free(C);
}

// gemm_NEON_2x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
free(C);
}

// gemm_NEON_2x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
free(C);
}

// gemm_NEON_2x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
free(C);
}

// gemm_NEON_2x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 2] @DRAM
// )
void gemm_NEON_2x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
free(C);
}

// gemm_NEON_2x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 2] @DRAM
// )
void gemm_NEON_2x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
free(C);
}

// gemm_NEON_2x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 2] @DRAM
// )
void gemm_NEON_2x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
free(C);
}

// gemm_NEON_2x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 2] @DRAM
// )
void gemm_NEON_2x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
free(C);
}

// gemm_NEON_2x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 2] @DRAM
// )
void gemm_NEON_2x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
free(C);
}

// gemm_NEON_2x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 2] @DRAM
// )
void gemm_NEON_2x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
free(C);
}

// gemm_NEON_2x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 2] @DRAM
// )
void gemm_NEON_2x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
free(C);
}

// gemm_NEON_2x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 2] @DRAM
// )
void gemm_NEON_2x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
free(C);
}

// gemm_NEON_2x17_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 2] @DRAM
// )
void gemm_NEON_2x17_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
free(C);
}

// gemm_NEON_2x17_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 2] @DRAM
// )
void gemm_NEON_2x17_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
free(C);
}

// gemm_NEON_2x18_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 2] @DRAM
// )
void gemm_NEON_2x18_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[17 * ldci] = C[68];
Ci[17 * ldci + 1] = C[69];
free(C);
}

// gemm_NEON_2x18_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 2] @DRAM
// )
void gemm_NEON_2x18_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[17 * ldci] += C[68];
Ci[17 * ldci + 1] += C[69];
free(C);
}

// gemm_NEON_2x19_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 2] @DRAM
// )
void gemm_NEON_2x19_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[17 * ldci] = C[68];
Ci[17 * ldci + 1] = C[69];
Ci[18 * ldci] = C[72];
Ci[18 * ldci + 1] = C[73];
free(C);
}

// gemm_NEON_2x19_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 2] @DRAM
// )
void gemm_NEON_2x19_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[17 * ldci] += C[68];
Ci[17 * ldci + 1] += C[69];
Ci[18 * ldci] += C[72];
Ci[18 * ldci + 1] += C[73];
free(C);
}

// gemm_NEON_2x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_2x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
C_reg_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
free(C);
}

// gemm_NEON_2x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_2x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
C_reg_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
free(C);
}

// gemm_NEON_2x20_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 2] @DRAM
// )
void gemm_NEON_2x20_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[17 * ldci] = C[68];
Ci[17 * ldci + 1] = C[69];
Ci[18 * ldci] = C[72];
Ci[18 * ldci + 1] = C[73];
Ci[19 * ldci] = C[76];
Ci[19 * ldci + 1] = C[77];
free(C);
}

// gemm_NEON_2x20_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 2] @DRAM
// )
void gemm_NEON_2x20_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[17 * ldci] += C[68];
Ci[17 * ldci + 1] += C[69];
Ci[18 * ldci] += C[72];
Ci[18 * ldci + 1] += C[73];
Ci[19 * ldci] += C[76];
Ci[19 * ldci + 1] += C[77];
free(C);
}

// gemm_NEON_2x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_2x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
free(C);
}

// gemm_NEON_2x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_2x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
free(C);
}

// gemm_NEON_2x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_2x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
free(C);
}

// gemm_NEON_2x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_2x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
free(C);
}

// gemm_NEON_2x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_2x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
free(C);
}

// gemm_NEON_2x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_2x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
free(C);
}

// gemm_NEON_2x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
free(C);
}

// gemm_NEON_2x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
free(C);
}

// gemm_NEON_2x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
free(C);
}

// gemm_NEON_2x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
free(C);
}

// gemm_NEON_2x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
free(C);
}

// gemm_NEON_2x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
free(C);
}

// gemm_NEON_2x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
free(C);
}

// gemm_NEON_2x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
free(C);
}

// gemm_NEON_2x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
free(C);
}

// gemm_NEON_2x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
free(C);
}

// gemm_NEON_3x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
free(C);
}

// gemm_NEON_3x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
free(C);
}

// gemm_NEON_3x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
free(C);
}

// gemm_NEON_3x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
free(C);
}

// gemm_NEON_3x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
free(C);
}

// gemm_NEON_3x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
free(C);
}

// gemm_NEON_3x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 3] @DRAM
// )
void gemm_NEON_3x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
free(C);
}

// gemm_NEON_3x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 3] @DRAM
// )
void gemm_NEON_3x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
free(C);
}

// gemm_NEON_3x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 3] @DRAM
// )
void gemm_NEON_3x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
free(C);
}

// gemm_NEON_3x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 3] @DRAM
// )
void gemm_NEON_3x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
free(C);
}

// gemm_NEON_3x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 3] @DRAM
// )
void gemm_NEON_3x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
free(C);
}

// gemm_NEON_3x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 3] @DRAM
// )
void gemm_NEON_3x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
free(C);
}

// gemm_NEON_3x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 3] @DRAM
// )
void gemm_NEON_3x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[15 * ldci + 2] = C[62];
free(C);
}

// gemm_NEON_3x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][16, 3] @DRAM
// )
void gemm_NEON_3x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[15 * ldci + 2] += C[62];
free(C);
}

// gemm_NEON_3x17_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 3] @DRAM
// )
void gemm_NEON_3x17_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[15 * ldci + 2] = C[62];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[16 * ldci + 2] = C[66];
free(C);
}

// gemm_NEON_3x17_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 3] @DRAM
// )
void gemm_NEON_3x17_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[15 * ldci + 2] += C[62];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[16 * ldci + 2] += C[66];
free(C);
}

// gemm_NEON_3x18_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 3] @DRAM
// )
void gemm_NEON_3x18_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[15 * ldci + 2] = C[62];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[16 * ldci + 2] = C[66];
Ci[17 * ldci] = C[68];
Ci[17 * ldci + 1] = C[69];
Ci[17 * ldci + 2] = C[70];
free(C);
}

// gemm_NEON_3x18_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 3] @DRAM
// )
void gemm_NEON_3x18_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[15 * ldci + 2] += C[62];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[16 * ldci + 2] += C[66];
Ci[17 * ldci] += C[68];
Ci[17 * ldci + 1] += C[69];
Ci[17 * ldci + 2] += C[70];
free(C);
}

// gemm_NEON_3x19_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 3] @DRAM
// )
void gemm_NEON_3x19_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[15 * ldci + 2] = C[62];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[16 * ldci + 2] = C[66];
Ci[17 * ldci] = C[68];
Ci[17 * ldci + 1] = C[69];
Ci[17 * ldci + 2] = C[70];
Ci[18 * ldci] = C[72];
Ci[18 * ldci + 1] = C[73];
Ci[18 * ldci + 2] = C[74];
free(C);
}

// gemm_NEON_3x19_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 3] @DRAM
// )
void gemm_NEON_3x19_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[15 * ldci + 2] += C[62];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[16 * ldci + 2] += C[66];
Ci[17 * ldci] += C[68];
Ci[17 * ldci + 1] += C[69];
Ci[17 * ldci + 2] += C[70];
Ci[18 * ldci] += C[72];
Ci[18 * ldci + 1] += C[73];
Ci[18 * ldci + 2] += C[74];
free(C);
}

// gemm_NEON_3x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_3x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
C_reg_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
free(C);
}

// gemm_NEON_3x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_3x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
C_reg_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
free(C);
}

// gemm_NEON_3x20_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 3] @DRAM
// )
void gemm_NEON_3x20_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[15 * ldci + 2] = C[62];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[16 * ldci + 2] = C[66];
Ci[17 * ldci] = C[68];
Ci[17 * ldci + 1] = C[69];
Ci[17 * ldci + 2] = C[70];
Ci[18 * ldci] = C[72];
Ci[18 * ldci + 1] = C[73];
Ci[18 * ldci + 2] = C[74];
Ci[19 * ldci] = C[76];
Ci[19 * ldci + 1] = C[77];
Ci[19 * ldci + 2] = C[78];
free(C);
}

// gemm_NEON_3x20_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][20, 3] @DRAM
// )
void gemm_NEON_3x20_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
float32x4_t C_reg_12;
float32x4_t C_reg_13;
float32x4_t C_reg_14;
float32x4_t C_reg_15;
float32x4_t C_reg_16;
float32x4_t C_reg_17;
float32x4_t C_reg_18;
float32x4_t C_reg_19;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
C_reg_12 = vmovq_n_f32(0.0f);
C_reg_13 = vmovq_n_f32(0.0f);
C_reg_14 = vmovq_n_f32(0.0f);
C_reg_15 = vmovq_n_f32(0.0f);
C_reg_16 = vmovq_n_f32(0.0f);
C_reg_17 = vmovq_n_f32(0.0f);
C_reg_18 = vmovq_n_f32(0.0f);
C_reg_19 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
  C_reg_12 = vfmaq_laneq_f32(C_reg_12, A_reg, B_reg_3, (0));
  C_reg_13 = vfmaq_laneq_f32(C_reg_13, A_reg, B_reg_3, (1));
  C_reg_14 = vfmaq_laneq_f32(C_reg_14, A_reg, B_reg_3, (2));
  C_reg_15 = vfmaq_laneq_f32(C_reg_15, A_reg, B_reg_3, (3));
  C_reg_16 = vfmaq_laneq_f32(C_reg_16, A_reg, B_reg_4, (0));
  C_reg_17 = vfmaq_laneq_f32(C_reg_17, A_reg, B_reg_4, (1));
  C_reg_18 = vfmaq_laneq_f32(C_reg_18, A_reg, B_reg_4, (2));
  C_reg_19 = vfmaq_laneq_f32(C_reg_19, A_reg, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
vst1q_f32(&C[(12) * 4], C_reg_12);
vst1q_f32(&C[(13) * 4], C_reg_13);
vst1q_f32(&C[(14) * 4], C_reg_14);
vst1q_f32(&C[(15) * 4], C_reg_15);
vst1q_f32(&C[(16) * 4], C_reg_16);
vst1q_f32(&C[(17) * 4], C_reg_17);
vst1q_f32(&C[(18) * 4], C_reg_18);
vst1q_f32(&C[(19) * 4], C_reg_19);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[15 * ldci + 2] += C[62];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[16 * ldci + 2] += C[66];
Ci[17 * ldci] += C[68];
Ci[17 * ldci + 1] += C[69];
Ci[17 * ldci + 2] += C[70];
Ci[18 * ldci] += C[72];
Ci[18 * ldci + 1] += C[73];
Ci[18 * ldci + 2] += C[74];
Ci[19 * ldci] += C[76];
Ci[19 * ldci + 1] += C[77];
Ci[19 * ldci + 2] += C[78];
free(C);
}

// gemm_NEON_3x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_3x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
free(C);
}

// gemm_NEON_3x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_3x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
free(C);
}

// gemm_NEON_3x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_3x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
free(C);
}

// gemm_NEON_3x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_3x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
free(C);
}

// gemm_NEON_3x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_3x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
free(C);
}

// gemm_NEON_3x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 4] @DRAM
// )
void gemm_NEON_3x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
free(C);
}

// gemm_NEON_3x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
free(C);
}

// gemm_NEON_3x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
free(C);
}

// gemm_NEON_3x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
free(C);
}

// gemm_NEON_3x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
free(C);
}

// gemm_NEON_3x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
free(C);
}

// gemm_NEON_3x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
free(C);
}

// gemm_NEON_3x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
free(C);
}

// gemm_NEON_3x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
free(C);
}

// gemm_NEON_3x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
free(C);
}

// gemm_NEON_3x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
float32x4_t C_reg_4;
float32x4_t C_reg_5;
float32x4_t C_reg_6;
float32x4_t C_reg_7;
float32x4_t C_reg_8;
float32x4_t C_reg_9;
float32x4_t C_reg_10;
float32x4_t C_reg_11;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
C_reg_4 = vmovq_n_f32(0.0f);
C_reg_5 = vmovq_n_f32(0.0f);
C_reg_6 = vmovq_n_f32(0.0f);
C_reg_7 = vmovq_n_f32(0.0f);
C_reg_8 = vmovq_n_f32(0.0f);
C_reg_9 = vmovq_n_f32(0.0f);
C_reg_10 = vmovq_n_f32(0.0f);
C_reg_11 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg_0, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg_0, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg_0, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg_0, (3));
  C_reg_4 = vfmaq_laneq_f32(C_reg_4, A_reg, B_reg_1, (0));
  C_reg_5 = vfmaq_laneq_f32(C_reg_5, A_reg, B_reg_1, (1));
  C_reg_6 = vfmaq_laneq_f32(C_reg_6, A_reg, B_reg_1, (2));
  C_reg_7 = vfmaq_laneq_f32(C_reg_7, A_reg, B_reg_1, (3));
  C_reg_8 = vfmaq_laneq_f32(C_reg_8, A_reg, B_reg_2, (0));
  C_reg_9 = vfmaq_laneq_f32(C_reg_9, A_reg, B_reg_2, (1));
  C_reg_10 = vfmaq_laneq_f32(C_reg_10, A_reg, B_reg_2, (2));
  C_reg_11 = vfmaq_laneq_f32(C_reg_11, A_reg, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
vst1q_f32(&C[(3) * 4], C_reg_3);
vst1q_f32(&C[(4) * 4], C_reg_4);
vst1q_f32(&C[(5) * 4], C_reg_5);
vst1q_f32(&C[(6) * 4], C_reg_6);
vst1q_f32(&C[(7) * 4], C_reg_7);
vst1q_f32(&C[(8) * 4], C_reg_8);
vst1q_f32(&C[(9) * 4], C_reg_9);
vst1q_f32(&C[(10) * 4], C_reg_10);
vst1q_f32(&C[(11) * 4], C_reg_11);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
free(C);
}

// gemm_NEON_4x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[9 * ldci + 3] = C[39];
free(C);
}

// gemm_NEON_4x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[9 * ldci + 3] += C[39];
free(C);
}

// gemm_NEON_4x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[9 * ldci + 3] = C[39];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[10 * ldci + 3] = C[43];
free(C);
}

// gemm_NEON_4x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[9 * ldci + 3] += C[39];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[10 * ldci + 3] += C[43];
free(C);
}

// gemm_NEON_4x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + (8) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(8) * (ldc)], C_reg_8_0);
vst1q_f32(&C[(9) * (ldc)], C_reg_9_0);
vst1q_f32(&C[(10) * (ldc)], C_reg_10_0);
vst1q_f32(&C[(11) * (ldc)], C_reg_11_0);
}

// gemm_NEON_4x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_4_0 = vld1q_f32(&C[(4) * (ldc)]);
C_reg_5_0 = vld1q_f32(&C[(5) * (ldc)]);
C_reg_6_0 = vld1q_f32(&C[(6) * (ldc)]);
C_reg_7_0 = vld1q_f32(&C[(7) * (ldc)]);
C_reg_8_0 = vld1q_f32(&C[(8) * (ldc)]);
C_reg_9_0 = vld1q_f32(&C[(9) * (ldc)]);
C_reg_10_0 = vld1q_f32(&C[(10) * (ldc)]);
C_reg_11_0 = vld1q_f32(&C[(11) * (ldc)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + (8) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(8) * (ldc)], C_reg_8_0);
vst1q_f32(&C[(9) * (ldc)], C_reg_9_0);
vst1q_f32(&C[(10) * (ldc)], C_reg_10_0);
vst1q_f32(&C[(11) * (ldc)], C_reg_11_0);
}

// gemm_NEON_4x13_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 4] @DRAM
// )
void gemm_NEON_4x13_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[9 * ldci + 3] = C[39];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[10 * ldci + 3] = C[43];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[11 * ldci + 3] = C[47];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[12 * ldci + 3] = C[51];
free(C);
}

// gemm_NEON_4x13_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][13, 4] @DRAM
// )
void gemm_NEON_4x13_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[9 * ldci + 3] += C[39];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[10 * ldci + 3] += C[43];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[11 * ldci + 3] += C[47];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[12 * ldci + 3] += C[51];
free(C);
}

// gemm_NEON_4x14_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 4] @DRAM
// )
void gemm_NEON_4x14_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[9 * ldci + 3] = C[39];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[10 * ldci + 3] = C[43];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[11 * ldci + 3] = C[47];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[12 * ldci + 3] = C[51];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[13 * ldci + 3] = C[55];
free(C);
}

// gemm_NEON_4x14_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][14, 4] @DRAM
// )
void gemm_NEON_4x14_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[9 * ldci + 3] += C[39];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[10 * ldci + 3] += C[43];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[11 * ldci + 3] += C[47];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[12 * ldci + 3] += C[51];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[13 * ldci + 3] += C[55];
free(C);
}

// gemm_NEON_4x15_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 4] @DRAM
// )
void gemm_NEON_4x15_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[9 * ldci + 3] = C[39];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[10 * ldci + 3] = C[43];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[11 * ldci + 3] = C[47];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[12 * ldci + 3] = C[51];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[13 * ldci + 3] = C[55];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[14 * ldci + 3] = C[59];
free(C);
}

// gemm_NEON_4x15_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][15, 4] @DRAM
// )
void gemm_NEON_4x15_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float *C = (float*) malloc(16 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[9 * ldci + 3] += C[39];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[10 * ldci + 3] += C[43];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[11 * ldci + 3] += C[47];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[12 * ldci + 3] += C[51];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[13 * ldci + 3] += C[55];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[14 * ldci + 3] += C[59];
free(C);
}

// gemm_NEON_4x16_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_NEON_4x16_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + (8) * (1)]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + (12) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(8) * (ldc)], C_reg_8_0);
vst1q_f32(&C[(9) * (ldc)], C_reg_9_0);
vst1q_f32(&C[(10) * (ldc)], C_reg_10_0);
vst1q_f32(&C[(11) * (ldc)], C_reg_11_0);
vst1q_f32(&C[(12) * (ldc)], C_reg_12_0);
vst1q_f32(&C[(13) * (ldc)], C_reg_13_0);
vst1q_f32(&C[(14) * (ldc)], C_reg_14_0);
vst1q_f32(&C[(15) * (ldc)], C_reg_15_0);
}

// gemm_NEON_4x16_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 16] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][16, 4] @DRAM
// )
void gemm_NEON_4x16_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_4_0 = vld1q_f32(&C[(4) * (ldc)]);
C_reg_5_0 = vld1q_f32(&C[(5) * (ldc)]);
C_reg_6_0 = vld1q_f32(&C[(6) * (ldc)]);
C_reg_7_0 = vld1q_f32(&C[(7) * (ldc)]);
C_reg_8_0 = vld1q_f32(&C[(8) * (ldc)]);
C_reg_9_0 = vld1q_f32(&C[(9) * (ldc)]);
C_reg_10_0 = vld1q_f32(&C[(10) * (ldc)]);
C_reg_11_0 = vld1q_f32(&C[(11) * (ldc)]);
C_reg_12_0 = vld1q_f32(&C[(12) * (ldc)]);
C_reg_13_0 = vld1q_f32(&C[(13) * (ldc)]);
C_reg_14_0 = vld1q_f32(&C[(14) * (ldc)]);
C_reg_15_0 = vld1q_f32(&C[(15) * (ldc)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + (8) * (1)]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + (12) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(8) * (ldc)], C_reg_8_0);
vst1q_f32(&C[(9) * (ldc)], C_reg_9_0);
vst1q_f32(&C[(10) * (ldc)], C_reg_10_0);
vst1q_f32(&C[(11) * (ldc)], C_reg_11_0);
vst1q_f32(&C[(12) * (ldc)], C_reg_12_0);
vst1q_f32(&C[(13) * (ldc)], C_reg_13_0);
vst1q_f32(&C[(14) * (ldc)], C_reg_14_0);
vst1q_f32(&C[(15) * (ldc)], C_reg_15_0);
}

// gemm_NEON_4x17_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 4] @DRAM
// )
void gemm_NEON_4x17_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
C_reg_16_0 = vmovq_n_f32(0.0f);
C_reg_17_0 = vmovq_n_f32(0.0f);
C_reg_18_0 = vmovq_n_f32(0.0f);
C_reg_19_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
vst1q_f32(&C[(16) * 4], C_reg_16_0);
vst1q_f32(&C[(17) * 4], C_reg_17_0);
vst1q_f32(&C[(18) * 4], C_reg_18_0);
vst1q_f32(&C[(19) * 4], C_reg_19_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[9 * ldci + 3] = C[39];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[10 * ldci + 3] = C[43];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[11 * ldci + 3] = C[47];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[12 * ldci + 3] = C[51];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[13 * ldci + 3] = C[55];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[14 * ldci + 3] = C[59];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[15 * ldci + 2] = C[62];
Ci[15 * ldci + 3] = C[63];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[16 * ldci + 2] = C[66];
Ci[16 * ldci + 3] = C[67];
free(C);
}

// gemm_NEON_4x17_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][17, 4] @DRAM
// )
void gemm_NEON_4x17_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
C_reg_16_0 = vmovq_n_f32(0.0f);
C_reg_17_0 = vmovq_n_f32(0.0f);
C_reg_18_0 = vmovq_n_f32(0.0f);
C_reg_19_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
vst1q_f32(&C[(16) * 4], C_reg_16_0);
vst1q_f32(&C[(17) * 4], C_reg_17_0);
vst1q_f32(&C[(18) * 4], C_reg_18_0);
vst1q_f32(&C[(19) * 4], C_reg_19_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[9 * ldci + 3] += C[39];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[10 * ldci + 3] += C[43];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[11 * ldci + 3] += C[47];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[12 * ldci + 3] += C[51];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[13 * ldci + 3] += C[55];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[14 * ldci + 3] += C[59];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[15 * ldci + 2] += C[62];
Ci[15 * ldci + 3] += C[63];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[16 * ldci + 2] += C[66];
Ci[16 * ldci + 3] += C[67];
free(C);
}

// gemm_NEON_4x18_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 4] @DRAM
// )
void gemm_NEON_4x18_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
C_reg_16_0 = vmovq_n_f32(0.0f);
C_reg_17_0 = vmovq_n_f32(0.0f);
C_reg_18_0 = vmovq_n_f32(0.0f);
C_reg_19_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
vst1q_f32(&C[(16) * 4], C_reg_16_0);
vst1q_f32(&C[(17) * 4], C_reg_17_0);
vst1q_f32(&C[(18) * 4], C_reg_18_0);
vst1q_f32(&C[(19) * 4], C_reg_19_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[9 * ldci + 3] = C[39];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[10 * ldci + 3] = C[43];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[11 * ldci + 3] = C[47];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[12 * ldci + 3] = C[51];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[13 * ldci + 3] = C[55];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[14 * ldci + 3] = C[59];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[15 * ldci + 2] = C[62];
Ci[15 * ldci + 3] = C[63];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[16 * ldci + 2] = C[66];
Ci[16 * ldci + 3] = C[67];
Ci[17 * ldci] = C[68];
Ci[17 * ldci + 1] = C[69];
Ci[17 * ldci + 2] = C[70];
Ci[17 * ldci + 3] = C[71];
free(C);
}

// gemm_NEON_4x18_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][18, 4] @DRAM
// )
void gemm_NEON_4x18_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
C_reg_16_0 = vmovq_n_f32(0.0f);
C_reg_17_0 = vmovq_n_f32(0.0f);
C_reg_18_0 = vmovq_n_f32(0.0f);
C_reg_19_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
vst1q_f32(&C[(16) * 4], C_reg_16_0);
vst1q_f32(&C[(17) * 4], C_reg_17_0);
vst1q_f32(&C[(18) * 4], C_reg_18_0);
vst1q_f32(&C[(19) * 4], C_reg_19_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[9 * ldci + 3] += C[39];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[10 * ldci + 3] += C[43];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[11 * ldci + 3] += C[47];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[12 * ldci + 3] += C[51];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[13 * ldci + 3] += C[55];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[14 * ldci + 3] += C[59];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[15 * ldci + 2] += C[62];
Ci[15 * ldci + 3] += C[63];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[16 * ldci + 2] += C[66];
Ci[16 * ldci + 3] += C[67];
Ci[17 * ldci] += C[68];
Ci[17 * ldci + 1] += C[69];
Ci[17 * ldci + 2] += C[70];
Ci[17 * ldci + 3] += C[71];
free(C);
}

// gemm_NEON_4x19_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 4] @DRAM
// )
void gemm_NEON_4x19_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
C_reg_16_0 = vmovq_n_f32(0.0f);
C_reg_17_0 = vmovq_n_f32(0.0f);
C_reg_18_0 = vmovq_n_f32(0.0f);
C_reg_19_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
vst1q_f32(&C[(16) * 4], C_reg_16_0);
vst1q_f32(&C[(17) * 4], C_reg_17_0);
vst1q_f32(&C[(18) * 4], C_reg_18_0);
vst1q_f32(&C[(19) * 4], C_reg_19_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
Ci[9 * ldci] = C[36];
Ci[9 * ldci + 1] = C[37];
Ci[9 * ldci + 2] = C[38];
Ci[9 * ldci + 3] = C[39];
Ci[10 * ldci] = C[40];
Ci[10 * ldci + 1] = C[41];
Ci[10 * ldci + 2] = C[42];
Ci[10 * ldci + 3] = C[43];
Ci[11 * ldci] = C[44];
Ci[11 * ldci + 1] = C[45];
Ci[11 * ldci + 2] = C[46];
Ci[11 * ldci + 3] = C[47];
Ci[12 * ldci] = C[48];
Ci[12 * ldci + 1] = C[49];
Ci[12 * ldci + 2] = C[50];
Ci[12 * ldci + 3] = C[51];
Ci[13 * ldci] = C[52];
Ci[13 * ldci + 1] = C[53];
Ci[13 * ldci + 2] = C[54];
Ci[13 * ldci + 3] = C[55];
Ci[14 * ldci] = C[56];
Ci[14 * ldci + 1] = C[57];
Ci[14 * ldci + 2] = C[58];
Ci[14 * ldci + 3] = C[59];
Ci[15 * ldci] = C[60];
Ci[15 * ldci + 1] = C[61];
Ci[15 * ldci + 2] = C[62];
Ci[15 * ldci + 3] = C[63];
Ci[16 * ldci] = C[64];
Ci[16 * ldci + 1] = C[65];
Ci[16 * ldci + 2] = C[66];
Ci[16 * ldci + 3] = C[67];
Ci[17 * ldci] = C[68];
Ci[17 * ldci + 1] = C[69];
Ci[17 * ldci + 2] = C[70];
Ci[17 * ldci + 3] = C[71];
Ci[18 * ldci] = C[72];
Ci[18 * ldci + 1] = C[73];
Ci[18 * ldci + 2] = C[74];
Ci[18 * ldci + 3] = C[75];
free(C);
}

// gemm_NEON_4x19_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][19, 4] @DRAM
// )
void gemm_NEON_4x19_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float *C = (float*) malloc(20 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
C_reg_16_0 = vmovq_n_f32(0.0f);
C_reg_17_0 = vmovq_n_f32(0.0f);
C_reg_18_0 = vmovq_n_f32(0.0f);
C_reg_19_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + 12]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + 16]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
vst1q_f32(&C[(12) * 4], C_reg_12_0);
vst1q_f32(&C[(13) * 4], C_reg_13_0);
vst1q_f32(&C[(14) * 4], C_reg_14_0);
vst1q_f32(&C[(15) * 4], C_reg_15_0);
vst1q_f32(&C[(16) * 4], C_reg_16_0);
vst1q_f32(&C[(17) * 4], C_reg_17_0);
vst1q_f32(&C[(18) * 4], C_reg_18_0);
vst1q_f32(&C[(19) * 4], C_reg_19_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
Ci[9 * ldci] += C[36];
Ci[9 * ldci + 1] += C[37];
Ci[9 * ldci + 2] += C[38];
Ci[9 * ldci + 3] += C[39];
Ci[10 * ldci] += C[40];
Ci[10 * ldci + 1] += C[41];
Ci[10 * ldci + 2] += C[42];
Ci[10 * ldci + 3] += C[43];
Ci[11 * ldci] += C[44];
Ci[11 * ldci + 1] += C[45];
Ci[11 * ldci + 2] += C[46];
Ci[11 * ldci + 3] += C[47];
Ci[12 * ldci] += C[48];
Ci[12 * ldci + 1] += C[49];
Ci[12 * ldci + 2] += C[50];
Ci[12 * ldci + 3] += C[51];
Ci[13 * ldci] += C[52];
Ci[13 * ldci + 1] += C[53];
Ci[13 * ldci + 2] += C[54];
Ci[13 * ldci + 3] += C[55];
Ci[14 * ldci] += C[56];
Ci[14 * ldci + 1] += C[57];
Ci[14 * ldci + 2] += C[58];
Ci[14 * ldci + 3] += C[59];
Ci[15 * ldci] += C[60];
Ci[15 * ldci + 1] += C[61];
Ci[15 * ldci + 2] += C[62];
Ci[15 * ldci + 3] += C[63];
Ci[16 * ldci] += C[64];
Ci[16 * ldci + 1] += C[65];
Ci[16 * ldci + 2] += C[66];
Ci[16 * ldci + 3] += C[67];
Ci[17 * ldci] += C[68];
Ci[17 * ldci + 1] += C[69];
Ci[17 * ldci + 2] += C[70];
Ci[17 * ldci + 3] += C[71];
Ci[18 * ldci] += C[72];
Ci[18 * ldci + 1] += C[73];
Ci[18 * ldci + 2] += C[74];
Ci[18 * ldci + 3] += C[75];
free(C);
}

// gemm_NEON_4x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
C_reg_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
free(C);
}

// gemm_NEON_4x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 1] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
C_reg_0 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
free(C);
}

// gemm_NEON_4x20_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 4] @DRAM
// )
void gemm_NEON_4x20_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_12_0 = vmovq_n_f32(0.0f);
C_reg_13_0 = vmovq_n_f32(0.0f);
C_reg_14_0 = vmovq_n_f32(0.0f);
C_reg_15_0 = vmovq_n_f32(0.0f);
C_reg_16_0 = vmovq_n_f32(0.0f);
C_reg_17_0 = vmovq_n_f32(0.0f);
C_reg_18_0 = vmovq_n_f32(0.0f);
C_reg_19_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + (8) * (1)]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + (12) * (1)]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + (16) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(8) * (ldc)], C_reg_8_0);
vst1q_f32(&C[(9) * (ldc)], C_reg_9_0);
vst1q_f32(&C[(10) * (ldc)], C_reg_10_0);
vst1q_f32(&C[(11) * (ldc)], C_reg_11_0);
vst1q_f32(&C[(12) * (ldc)], C_reg_12_0);
vst1q_f32(&C[(13) * (ldc)], C_reg_13_0);
vst1q_f32(&C[(14) * (ldc)], C_reg_14_0);
vst1q_f32(&C[(15) * (ldc)], C_reg_15_0);
vst1q_f32(&C[(16) * (ldc)], C_reg_16_0);
vst1q_f32(&C[(17) * (ldc)], C_reg_17_0);
vst1q_f32(&C[(18) * (ldc)], C_reg_18_0);
vst1q_f32(&C[(19) * (ldc)], C_reg_19_0);
}

// gemm_NEON_4x20_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 20] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][20, 4] @DRAM
// )
void gemm_NEON_4x20_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2, B_reg_3, B_reg_4;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
float32x4_t C_reg_12_0;
float32x4_t C_reg_13_0;
float32x4_t C_reg_14_0;
float32x4_t C_reg_15_0;
float32x4_t C_reg_16_0;
float32x4_t C_reg_17_0;
float32x4_t C_reg_18_0;
float32x4_t C_reg_19_0;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_4_0 = vld1q_f32(&C[(4) * (ldc)]);
C_reg_5_0 = vld1q_f32(&C[(5) * (ldc)]);
C_reg_6_0 = vld1q_f32(&C[(6) * (ldc)]);
C_reg_7_0 = vld1q_f32(&C[(7) * (ldc)]);
C_reg_8_0 = vld1q_f32(&C[(8) * (ldc)]);
C_reg_9_0 = vld1q_f32(&C[(9) * (ldc)]);
C_reg_10_0 = vld1q_f32(&C[(10) * (ldc)]);
C_reg_11_0 = vld1q_f32(&C[(11) * (ldc)]);
C_reg_12_0 = vld1q_f32(&C[(12) * (ldc)]);
C_reg_13_0 = vld1q_f32(&C[(13) * (ldc)]);
C_reg_14_0 = vld1q_f32(&C[(14) * (ldc)]);
C_reg_15_0 = vld1q_f32(&C[(15) * (ldc)]);
C_reg_16_0 = vld1q_f32(&C[(16) * (ldc)]);
C_reg_17_0 = vld1q_f32(&C[(17) * (ldc)]);
C_reg_18_0 = vld1q_f32(&C[(18) * (ldc)]);
C_reg_19_0 = vld1q_f32(&C[(19) * (ldc)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + (8) * (1)]);
  B_reg_3 = vld1q_f32(&B[(k) * (20) + (12) * (1)]);
  B_reg_4 = vld1q_f32(&B[(k) * (20) + (16) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_12_0 = vfmaq_laneq_f32(C_reg_12_0, A_reg_0, B_reg_3, (0));
  C_reg_13_0 = vfmaq_laneq_f32(C_reg_13_0, A_reg_0, B_reg_3, (1));
  C_reg_14_0 = vfmaq_laneq_f32(C_reg_14_0, A_reg_0, B_reg_3, (2));
  C_reg_15_0 = vfmaq_laneq_f32(C_reg_15_0, A_reg_0, B_reg_3, (3));
  C_reg_16_0 = vfmaq_laneq_f32(C_reg_16_0, A_reg_0, B_reg_4, (0));
  C_reg_17_0 = vfmaq_laneq_f32(C_reg_17_0, A_reg_0, B_reg_4, (1));
  C_reg_18_0 = vfmaq_laneq_f32(C_reg_18_0, A_reg_0, B_reg_4, (2));
  C_reg_19_0 = vfmaq_laneq_f32(C_reg_19_0, A_reg_0, B_reg_4, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(8) * (ldc)], C_reg_8_0);
vst1q_f32(&C[(9) * (ldc)], C_reg_9_0);
vst1q_f32(&C[(10) * (ldc)], C_reg_10_0);
vst1q_f32(&C[(11) * (ldc)], C_reg_11_0);
vst1q_f32(&C[(12) * (ldc)], C_reg_12_0);
vst1q_f32(&C[(13) * (ldc)], C_reg_13_0);
vst1q_f32(&C[(14) * (ldc)], C_reg_14_0);
vst1q_f32(&C[(15) * (ldc)], C_reg_15_0);
vst1q_f32(&C[(16) * (ldc)], C_reg_16_0);
vst1q_f32(&C[(17) * (ldc)], C_reg_17_0);
vst1q_f32(&C[(18) * (ldc)], C_reg_18_0);
vst1q_f32(&C[(19) * (ldc)], C_reg_19_0);
}

// gemm_NEON_4x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
free(C);
}

// gemm_NEON_4x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 2] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
free(C);
}

// gemm_NEON_4x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
free(C);
}

// gemm_NEON_4x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 3] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float *C = (float*) malloc(4 * 4 * sizeof(*C));
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[4], C_reg_1);
vst1q_f32(&C[(2) * 4], C_reg_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
free(C);
}

// gemm_NEON_4x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
C_reg_0 = vmovq_n_f32(0.0f);
C_reg_1 = vmovq_n_f32(0.0f);
C_reg_2 = vmovq_n_f32(0.0f);
C_reg_3 = vmovq_n_f32(0.0f);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[ldc], C_reg_1);
vst1q_f32(&C[(2) * (ldc)], C_reg_2);
vst1q_f32(&C[(3) * (ldc)], C_reg_3);
}

// gemm_NEON_4x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t C_reg_0;
float32x4_t C_reg_1;
float32x4_t C_reg_2;
float32x4_t C_reg_3;
C_reg_0 = vld1q_f32(&C[0]);
C_reg_1 = vld1q_f32(&C[ldc]);
C_reg_2 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_3 = vld1q_f32(&C[(3) * (ldc)]);
float32x4_t A_reg;
float32x4_t B_reg;
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg = vld1q_f32(&A[(k) * (8)]);
  B_reg = vld1q_f32(&B[(k) * (20)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
  C_reg_1 = vfmaq_laneq_f32(C_reg_1, A_reg, B_reg, (1));
  C_reg_2 = vfmaq_laneq_f32(C_reg_2, A_reg, B_reg, (2));
  C_reg_3 = vfmaq_laneq_f32(C_reg_3, A_reg, B_reg, (3));
}
vst1q_f32(&C[0], C_reg_0);
vst1q_f32(&C[ldc], C_reg_1);
vst1q_f32(&C[(2) * (ldc)], C_reg_2);
vst1q_f32(&C[(3) * (ldc)], C_reg_3);
}

// gemm_NEON_4x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
free(C);
}

// gemm_NEON_4x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
free(C);
}

// gemm_NEON_4x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
free(C);
}

// gemm_NEON_4x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
free(C);
}

// gemm_NEON_4x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
free(C);
}

// gemm_NEON_4x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
free(C);
}

// gemm_NEON_4x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
}

// gemm_NEON_4x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_4_0 = vld1q_f32(&C[(4) * (ldc)]);
C_reg_5_0 = vld1q_f32(&C[(5) * (ldc)]);
C_reg_6_0 = vld1q_f32(&C[(6) * (ldc)]);
C_reg_7_0 = vld1q_f32(&C[(7) * (ldc)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
}

// gemm_NEON_4x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[ldci] = C[4];
Ci[ldci + 1] = C[5];
Ci[ldci + 2] = C[6];
Ci[ldci + 3] = C[7];
Ci[2 * ldci] = C[8];
Ci[2 * ldci + 1] = C[9];
Ci[2 * ldci + 2] = C[10];
Ci[2 * ldci + 3] = C[11];
Ci[3 * ldci] = C[12];
Ci[3 * ldci + 1] = C[13];
Ci[3 * ldci + 2] = C[14];
Ci[3 * ldci + 3] = C[15];
Ci[4 * ldci] = C[16];
Ci[4 * ldci + 1] = C[17];
Ci[4 * ldci + 2] = C[18];
Ci[4 * ldci + 3] = C[19];
Ci[5 * ldci] = C[20];
Ci[5 * ldci + 1] = C[21];
Ci[5 * ldci + 2] = C[22];
Ci[5 * ldci + 3] = C[23];
Ci[6 * ldci] = C[24];
Ci[6 * ldci + 1] = C[25];
Ci[6 * ldci + 2] = C[26];
Ci[6 * ldci + 3] = C[27];
Ci[7 * ldci] = C[28];
Ci[7 * ldci + 1] = C[29];
Ci[7 * ldci + 2] = C[30];
Ci[7 * ldci + 3] = C[31];
Ci[8 * ldci] = C[32];
Ci[8 * ldci + 1] = C[33];
Ci[8 * ldci + 2] = C[34];
Ci[8 * ldci + 3] = C[35];
free(C);
}

// gemm_NEON_4x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 4] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 4 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_1_0;
float32x4_t C_reg_2_0;
float32x4_t C_reg_3_0;
float32x4_t C_reg_4_0;
float32x4_t C_reg_5_0;
float32x4_t C_reg_6_0;
float32x4_t C_reg_7_0;
float32x4_t C_reg_8_0;
float32x4_t C_reg_9_0;
float32x4_t C_reg_10_0;
float32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_1_0);
vst1q_f32(&C[(2) * 4], C_reg_2_0);
vst1q_f32(&C[(3) * 4], C_reg_3_0);
vst1q_f32(&C[(4) * 4], C_reg_4_0);
vst1q_f32(&C[(5) * 4], C_reg_5_0);
vst1q_f32(&C[(6) * 4], C_reg_6_0);
vst1q_f32(&C[(7) * 4], C_reg_7_0);
vst1q_f32(&C[(8) * 4], C_reg_8_0);
vst1q_f32(&C[(9) * 4], C_reg_9_0);
vst1q_f32(&C[(10) * 4], C_reg_10_0);
vst1q_f32(&C[(11) * 4], C_reg_11_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[ldci] += C[4];
Ci[ldci + 1] += C[5];
Ci[ldci + 2] += C[6];
Ci[ldci + 3] += C[7];
Ci[2 * ldci] += C[8];
Ci[2 * ldci + 1] += C[9];
Ci[2 * ldci + 2] += C[10];
Ci[2 * ldci + 3] += C[11];
Ci[3 * ldci] += C[12];
Ci[3 * ldci + 1] += C[13];
Ci[3 * ldci + 2] += C[14];
Ci[3 * ldci + 3] += C[15];
Ci[4 * ldci] += C[16];
Ci[4 * ldci + 1] += C[17];
Ci[4 * ldci + 2] += C[18];
Ci[4 * ldci + 3] += C[19];
Ci[5 * ldci] += C[20];
Ci[5 * ldci + 1] += C[21];
Ci[5 * ldci + 2] += C[22];
Ci[5 * ldci + 3] += C[23];
Ci[6 * ldci] += C[24];
Ci[6 * ldci + 1] += C[25];
Ci[6 * ldci + 2] += C[26];
Ci[6 * ldci + 3] += C[27];
Ci[7 * ldci] += C[28];
Ci[7 * ldci + 1] += C[29];
Ci[7 * ldci + 2] += C[30];
Ci[7 * ldci + 3] += C[31];
Ci[8 * ldci] += C[32];
Ci[8 * ldci + 1] += C[33];
Ci[8 * ldci + 2] += C[34];
Ci[8 * ldci + 3] += C[35];
free(C);
}

// gemm_NEON_5x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 5] @DRAM
// )
void gemm_NEON_5x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
free(C);
}

// gemm_NEON_5x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 5] @DRAM
// )
void gemm_NEON_5x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
free(C);
}

// gemm_NEON_5x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 5] @DRAM
// )
void gemm_NEON_5x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[10 * ldci] = C[80];
Ci[10 * ldci + 1] = C[81];
Ci[10 * ldci + 2] = C[82];
Ci[10 * ldci + 3] = C[83];
Ci[10 * ldci + 4] = C[84];
free(C);
}

// gemm_NEON_5x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 5] @DRAM
// )
void gemm_NEON_5x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[10 * ldci] += C[80];
Ci[10 * ldci + 1] += C[81];
Ci[10 * ldci + 2] += C[82];
Ci[10 * ldci + 3] += C[83];
Ci[10 * ldci + 4] += C[84];
free(C);
}

// gemm_NEON_5x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 5] @DRAM
// )
void gemm_NEON_5x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[10 * ldci] = C[80];
Ci[10 * ldci + 1] = C[81];
Ci[10 * ldci + 2] = C[82];
Ci[10 * ldci + 3] = C[83];
Ci[10 * ldci + 4] = C[84];
Ci[11 * ldci] = C[88];
Ci[11 * ldci + 1] = C[89];
Ci[11 * ldci + 2] = C[90];
Ci[11 * ldci + 3] = C[91];
Ci[11 * ldci + 4] = C[92];
free(C);
}

// gemm_NEON_5x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 5] @DRAM
// )
void gemm_NEON_5x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[10 * ldci] += C[80];
Ci[10 * ldci + 1] += C[81];
Ci[10 * ldci + 2] += C[82];
Ci[10 * ldci + 3] += C[83];
Ci[10 * ldci + 4] += C[84];
Ci[11 * ldci] += C[88];
Ci[11 * ldci + 1] += C[89];
Ci[11 * ldci + 2] += C[90];
Ci[11 * ldci + 3] += C[91];
Ci[11 * ldci + 4] += C[92];
free(C);
}

// gemm_NEON_5x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 5] @DRAM
// )
void gemm_NEON_5x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
free(C);
}

// gemm_NEON_5x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 5] @DRAM
// )
void gemm_NEON_5x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
free(C);
}

// gemm_NEON_5x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 5] @DRAM
// )
void gemm_NEON_5x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_5x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 5] @DRAM
// )
void gemm_NEON_5x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_5x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 5] @DRAM
// )
void gemm_NEON_5x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_5x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 5] @DRAM
// )
void gemm_NEON_5x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_5x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 5] @DRAM
// )
void gemm_NEON_5x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_5x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 5] @DRAM
// )
void gemm_NEON_5x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_5x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 5] @DRAM
// )
void gemm_NEON_5x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_5x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 5] @DRAM
// )
void gemm_NEON_5x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_5x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 5] @DRAM
// )
void gemm_NEON_5x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_5x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 5] @DRAM
// )
void gemm_NEON_5x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_5x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 5] @DRAM
// )
void gemm_NEON_5x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_5x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 5] @DRAM
// )
void gemm_NEON_5x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_5x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 5] @DRAM
// )
void gemm_NEON_5x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_5x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 5] @DRAM
// )
void gemm_NEON_5x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_5x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 5] @DRAM
// )
void gemm_NEON_5x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
free(C);
}

// gemm_NEON_5x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 5] @DRAM
// )
void gemm_NEON_5x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
free(C);
}

// gemm_NEON_6x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 6] @DRAM
// )
void gemm_NEON_6x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[9 * ldci + 5] = C[77];
free(C);
}

// gemm_NEON_6x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 6] @DRAM
// )
void gemm_NEON_6x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[9 * ldci + 5] += C[77];
free(C);
}

// gemm_NEON_6x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 6] @DRAM
// )
void gemm_NEON_6x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[9 * ldci + 5] = C[77];
Ci[10 * ldci] = C[80];
Ci[10 * ldci + 1] = C[81];
Ci[10 * ldci + 2] = C[82];
Ci[10 * ldci + 3] = C[83];
Ci[10 * ldci + 4] = C[84];
Ci[10 * ldci + 5] = C[85];
free(C);
}

// gemm_NEON_6x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 6] @DRAM
// )
void gemm_NEON_6x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[9 * ldci + 5] += C[77];
Ci[10 * ldci] += C[80];
Ci[10 * ldci + 1] += C[81];
Ci[10 * ldci + 2] += C[82];
Ci[10 * ldci + 3] += C[83];
Ci[10 * ldci + 4] += C[84];
Ci[10 * ldci + 5] += C[85];
free(C);
}

// gemm_NEON_6x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 6] @DRAM
// )
void gemm_NEON_6x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[9 * ldci + 5] = C[77];
Ci[10 * ldci] = C[80];
Ci[10 * ldci + 1] = C[81];
Ci[10 * ldci + 2] = C[82];
Ci[10 * ldci + 3] = C[83];
Ci[10 * ldci + 4] = C[84];
Ci[10 * ldci + 5] = C[85];
Ci[11 * ldci] = C[88];
Ci[11 * ldci + 1] = C[89];
Ci[11 * ldci + 2] = C[90];
Ci[11 * ldci + 3] = C[91];
Ci[11 * ldci + 4] = C[92];
Ci[11 * ldci + 5] = C[93];
free(C);
}

// gemm_NEON_6x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 6] @DRAM
// )
void gemm_NEON_6x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[9 * ldci + 5] += C[77];
Ci[10 * ldci] += C[80];
Ci[10 * ldci + 1] += C[81];
Ci[10 * ldci + 2] += C[82];
Ci[10 * ldci + 3] += C[83];
Ci[10 * ldci + 4] += C[84];
Ci[10 * ldci + 5] += C[85];
Ci[11 * ldci] += C[88];
Ci[11 * ldci + 1] += C[89];
Ci[11 * ldci + 2] += C[90];
Ci[11 * ldci + 3] += C[91];
Ci[11 * ldci + 4] += C[92];
Ci[11 * ldci + 5] += C[93];
free(C);
}

// gemm_NEON_6x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 6] @DRAM
// )
void gemm_NEON_6x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
free(C);
}

// gemm_NEON_6x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 6] @DRAM
// )
void gemm_NEON_6x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
free(C);
}

// gemm_NEON_6x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 6] @DRAM
// )
void gemm_NEON_6x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_6x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 6] @DRAM
// )
void gemm_NEON_6x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_6x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 6] @DRAM
// )
void gemm_NEON_6x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_6x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 6] @DRAM
// )
void gemm_NEON_6x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_6x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 6] @DRAM
// )
void gemm_NEON_6x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_6x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 6] @DRAM
// )
void gemm_NEON_6x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_6x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 6] @DRAM
// )
void gemm_NEON_6x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_6x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 6] @DRAM
// )
void gemm_NEON_6x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_6x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 6] @DRAM
// )
void gemm_NEON_6x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_6x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 6] @DRAM
// )
void gemm_NEON_6x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_6x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 6] @DRAM
// )
void gemm_NEON_6x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_6x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 6] @DRAM
// )
void gemm_NEON_6x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_6x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 6] @DRAM
// )
void gemm_NEON_6x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_6x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 6] @DRAM
// )
void gemm_NEON_6x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_6x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 6] @DRAM
// )
void gemm_NEON_6x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
free(C);
}

// gemm_NEON_6x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 6] @DRAM
// )
void gemm_NEON_6x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
free(C);
}

// gemm_NEON_7x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 7] @DRAM
// )
void gemm_NEON_7x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[8 * ldci + 6] = C[70];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[9 * ldci + 5] = C[77];
Ci[9 * ldci + 6] = C[78];
free(C);
}

// gemm_NEON_7x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 7] @DRAM
// )
void gemm_NEON_7x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[8 * ldci + 6] += C[70];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[9 * ldci + 5] += C[77];
Ci[9 * ldci + 6] += C[78];
free(C);
}

// gemm_NEON_7x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 7] @DRAM
// )
void gemm_NEON_7x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[8 * ldci + 6] = C[70];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[9 * ldci + 5] = C[77];
Ci[9 * ldci + 6] = C[78];
Ci[10 * ldci] = C[80];
Ci[10 * ldci + 1] = C[81];
Ci[10 * ldci + 2] = C[82];
Ci[10 * ldci + 3] = C[83];
Ci[10 * ldci + 4] = C[84];
Ci[10 * ldci + 5] = C[85];
Ci[10 * ldci + 6] = C[86];
free(C);
}

// gemm_NEON_7x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 7] @DRAM
// )
void gemm_NEON_7x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[8 * ldci + 6] += C[70];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[9 * ldci + 5] += C[77];
Ci[9 * ldci + 6] += C[78];
Ci[10 * ldci] += C[80];
Ci[10 * ldci + 1] += C[81];
Ci[10 * ldci + 2] += C[82];
Ci[10 * ldci + 3] += C[83];
Ci[10 * ldci + 4] += C[84];
Ci[10 * ldci + 5] += C[85];
Ci[10 * ldci + 6] += C[86];
free(C);
}

// gemm_NEON_7x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 7] @DRAM
// )
void gemm_NEON_7x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[8 * ldci + 6] = C[70];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[9 * ldci + 5] = C[77];
Ci[9 * ldci + 6] = C[78];
Ci[10 * ldci] = C[80];
Ci[10 * ldci + 1] = C[81];
Ci[10 * ldci + 2] = C[82];
Ci[10 * ldci + 3] = C[83];
Ci[10 * ldci + 4] = C[84];
Ci[10 * ldci + 5] = C[85];
Ci[10 * ldci + 6] = C[86];
Ci[11 * ldci] = C[88];
Ci[11 * ldci + 1] = C[89];
Ci[11 * ldci + 2] = C[90];
Ci[11 * ldci + 3] = C[91];
Ci[11 * ldci + 4] = C[92];
Ci[11 * ldci + 5] = C[93];
Ci[11 * ldci + 6] = C[94];
free(C);
}

// gemm_NEON_7x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][12, 7] @DRAM
// )
void gemm_NEON_7x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[8 * ldci + 6] += C[70];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[9 * ldci + 5] += C[77];
Ci[9 * ldci + 6] += C[78];
Ci[10 * ldci] += C[80];
Ci[10 * ldci + 1] += C[81];
Ci[10 * ldci + 2] += C[82];
Ci[10 * ldci + 3] += C[83];
Ci[10 * ldci + 4] += C[84];
Ci[10 * ldci + 5] += C[85];
Ci[10 * ldci + 6] += C[86];
Ci[11 * ldci] += C[88];
Ci[11 * ldci + 1] += C[89];
Ci[11 * ldci + 2] += C[90];
Ci[11 * ldci + 3] += C[91];
Ci[11 * ldci + 4] += C[92];
Ci[11 * ldci + 5] += C[93];
Ci[11 * ldci + 6] += C[94];
free(C);
}

// gemm_NEON_7x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 7] @DRAM
// )
void gemm_NEON_7x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
free(C);
}

// gemm_NEON_7x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 7] @DRAM
// )
void gemm_NEON_7x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
free(C);
}

// gemm_NEON_7x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 7] @DRAM
// )
void gemm_NEON_7x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_7x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 7] @DRAM
// )
void gemm_NEON_7x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_7x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 7] @DRAM
// )
void gemm_NEON_7x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_7x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 7] @DRAM
// )
void gemm_NEON_7x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_7x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 7] @DRAM
// )
void gemm_NEON_7x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_7x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 7] @DRAM
// )
void gemm_NEON_7x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_7x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 7] @DRAM
// )
void gemm_NEON_7x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_7x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 7] @DRAM
// )
void gemm_NEON_7x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_7x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 7] @DRAM
// )
void gemm_NEON_7x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_7x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 7] @DRAM
// )
void gemm_NEON_7x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_7x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 7] @DRAM
// )
void gemm_NEON_7x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_7x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 7] @DRAM
// )
void gemm_NEON_7x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_7x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 7] @DRAM
// )
void gemm_NEON_7x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_7x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][8, 7] @DRAM
// )
void gemm_NEON_7x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_7x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 7] @DRAM
// )
void gemm_NEON_7x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[8 * ldci + 6] = C[70];
free(C);
}

// gemm_NEON_7x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 7] @DRAM
// )
void gemm_NEON_7x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[8 * ldci + 6] += C[70];
free(C);
}

// gemm_NEON_8x10_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 8] @DRAM
// )
void gemm_NEON_8x10_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
Ci[7 * ldci + 2] = C[58];
Ci[7 * ldci + 3] = C[59];
Ci[7 * ldci + 4] = C[60];
Ci[7 * ldci + 5] = C[61];
Ci[7 * ldci + 6] = C[62];
Ci[7 * ldci + 7] = C[63];
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[8 * ldci + 6] = C[70];
Ci[8 * ldci + 7] = C[71];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[9 * ldci + 5] = C[77];
Ci[9 * ldci + 6] = C[78];
Ci[9 * ldci + 7] = C[79];
free(C);
}

// gemm_NEON_8x10_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][10, 8] @DRAM
// )
void gemm_NEON_8x10_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
Ci[7 * ldci + 2] += C[58];
Ci[7 * ldci + 3] += C[59];
Ci[7 * ldci + 4] += C[60];
Ci[7 * ldci + 5] += C[61];
Ci[7 * ldci + 6] += C[62];
Ci[7 * ldci + 7] += C[63];
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[8 * ldci + 6] += C[70];
Ci[8 * ldci + 7] += C[71];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[9 * ldci + 5] += C[77];
Ci[9 * ldci + 6] += C[78];
Ci[9 * ldci + 7] += C[79];
free(C);
}

// gemm_NEON_8x11_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 8] @DRAM
// )
void gemm_NEON_8x11_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
Ci[7 * ldci + 2] = C[58];
Ci[7 * ldci + 3] = C[59];
Ci[7 * ldci + 4] = C[60];
Ci[7 * ldci + 5] = C[61];
Ci[7 * ldci + 6] = C[62];
Ci[7 * ldci + 7] = C[63];
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[8 * ldci + 6] = C[70];
Ci[8 * ldci + 7] = C[71];
Ci[9 * ldci] = C[72];
Ci[9 * ldci + 1] = C[73];
Ci[9 * ldci + 2] = C[74];
Ci[9 * ldci + 3] = C[75];
Ci[9 * ldci + 4] = C[76];
Ci[9 * ldci + 5] = C[77];
Ci[9 * ldci + 6] = C[78];
Ci[9 * ldci + 7] = C[79];
Ci[10 * ldci] = C[80];
Ci[10 * ldci + 1] = C[81];
Ci[10 * ldci + 2] = C[82];
Ci[10 * ldci + 3] = C[83];
Ci[10 * ldci + 4] = C[84];
Ci[10 * ldci + 5] = C[85];
Ci[10 * ldci + 6] = C[86];
Ci[10 * ldci + 7] = C[87];
free(C);
}

// gemm_NEON_8x11_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][11, 8] @DRAM
// )
void gemm_NEON_8x11_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
Ci[7 * ldci + 2] += C[58];
Ci[7 * ldci + 3] += C[59];
Ci[7 * ldci + 4] += C[60];
Ci[7 * ldci + 5] += C[61];
Ci[7 * ldci + 6] += C[62];
Ci[7 * ldci + 7] += C[63];
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[8 * ldci + 6] += C[70];
Ci[8 * ldci + 7] += C[71];
Ci[9 * ldci] += C[72];
Ci[9 * ldci + 1] += C[73];
Ci[9 * ldci + 2] += C[74];
Ci[9 * ldci + 3] += C[75];
Ci[9 * ldci + 4] += C[76];
Ci[9 * ldci + 5] += C[77];
Ci[9 * ldci + 6] += C[78];
Ci[9 * ldci + 7] += C[79];
Ci[10 * ldci] += C[80];
Ci[10 * ldci + 1] += C[81];
Ci[10 * ldci + 2] += C[82];
Ci[10 * ldci + 3] += C[83];
Ci[10 * ldci + 4] += C[84];
Ci[10 * ldci + 5] += C[85];
Ci[10 * ldci + 6] += C[86];
Ci[10 * ldci + 7] += C[87];
free(C);
}

// gemm_NEON_8x12_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_NEON_8x12_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + (4) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + (8) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(4) * (ldc) + (4) * (1)], C_reg_4_1);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(5) * (ldc) + (4) * (1)], C_reg_5_1);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(6) * (ldc) + (4) * (1)], C_reg_6_1);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(7) * (ldc) + (4) * (1)], C_reg_7_1);
vst1q_f32(&C[(8) * (ldc)], C_reg_8_0);
vst1q_f32(&C[(8) * (ldc) + (4) * (1)], C_reg_8_1);
vst1q_f32(&C[(9) * (ldc)], C_reg_9_0);
vst1q_f32(&C[(9) * (ldc) + (4) * (1)], C_reg_9_1);
vst1q_f32(&C[(10) * (ldc)], C_reg_10_0);
vst1q_f32(&C[(10) * (ldc) + (4) * (1)], C_reg_10_1);
vst1q_f32(&C[(11) * (ldc)], C_reg_11_0);
vst1q_f32(&C[(11) * (ldc) + (4) * (1)], C_reg_11_1);
}

// gemm_NEON_8x12_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][12, 8] @DRAM
// )
void gemm_NEON_8x12_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_0_1 = vld1q_f32(&C[(4) * (1)]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_1_1 = vld1q_f32(&C[ldc + (4) * (1)]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_2_1 = vld1q_f32(&C[(2) * (ldc) + (4) * (1)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_3_1 = vld1q_f32(&C[(3) * (ldc) + (4) * (1)]);
C_reg_4_0 = vld1q_f32(&C[(4) * (ldc)]);
C_reg_4_1 = vld1q_f32(&C[(4) * (ldc) + (4) * (1)]);
C_reg_5_0 = vld1q_f32(&C[(5) * (ldc)]);
C_reg_5_1 = vld1q_f32(&C[(5) * (ldc) + (4) * (1)]);
C_reg_6_0 = vld1q_f32(&C[(6) * (ldc)]);
C_reg_6_1 = vld1q_f32(&C[(6) * (ldc) + (4) * (1)]);
C_reg_7_0 = vld1q_f32(&C[(7) * (ldc)]);
C_reg_7_1 = vld1q_f32(&C[(7) * (ldc) + (4) * (1)]);
C_reg_8_0 = vld1q_f32(&C[(8) * (ldc)]);
C_reg_8_1 = vld1q_f32(&C[(8) * (ldc) + (4) * (1)]);
C_reg_9_0 = vld1q_f32(&C[(9) * (ldc)]);
C_reg_9_1 = vld1q_f32(&C[(9) * (ldc) + (4) * (1)]);
C_reg_10_0 = vld1q_f32(&C[(10) * (ldc)]);
C_reg_10_1 = vld1q_f32(&C[(10) * (ldc) + (4) * (1)]);
C_reg_11_0 = vld1q_f32(&C[(11) * (ldc)]);
C_reg_11_1 = vld1q_f32(&C[(11) * (ldc) + (4) * (1)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + (4) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + (8) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(4) * (ldc) + (4) * (1)], C_reg_4_1);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(5) * (ldc) + (4) * (1)], C_reg_5_1);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(6) * (ldc) + (4) * (1)], C_reg_6_1);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(7) * (ldc) + (4) * (1)], C_reg_7_1);
vst1q_f32(&C[(8) * (ldc)], C_reg_8_0);
vst1q_f32(&C[(8) * (ldc) + (4) * (1)], C_reg_8_1);
vst1q_f32(&C[(9) * (ldc)], C_reg_9_0);
vst1q_f32(&C[(9) * (ldc) + (4) * (1)], C_reg_9_1);
vst1q_f32(&C[(10) * (ldc)], C_reg_10_0);
vst1q_f32(&C[(10) * (ldc) + (4) * (1)], C_reg_10_1);
vst1q_f32(&C[(11) * (ldc)], C_reg_11_0);
vst1q_f32(&C[(11) * (ldc) + (4) * (1)], C_reg_11_1);
}

// gemm_NEON_8x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 8] @DRAM
// )
void gemm_NEON_8x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_8x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 8] @DRAM
// )
void gemm_NEON_8x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_8x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 8] @DRAM
// )
void gemm_NEON_8x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_8x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 8] @DRAM
// )
void gemm_NEON_8x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_8x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 8] @DRAM
// )
void gemm_NEON_8x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_8x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 8] @DRAM
// )
void gemm_NEON_8x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
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

// gemm_NEON_8x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_NEON_8x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + (4) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
}

// gemm_NEON_8x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 8] @DRAM
// )
void gemm_NEON_8x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_0_1 = vld1q_f32(&C[(4) * (1)]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_1_1 = vld1q_f32(&C[ldc + (4) * (1)]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_2_1 = vld1q_f32(&C[(2) * (ldc) + (4) * (1)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_3_1 = vld1q_f32(&C[(3) * (ldc) + (4) * (1)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + (4) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
}

// gemm_NEON_8x5_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 8] @DRAM
// )
void gemm_NEON_8x5_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_8x5_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][5, 8] @DRAM
// )
void gemm_NEON_8x5_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_8x6_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 8] @DRAM
// )
void gemm_NEON_8x6_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_8x6_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][6, 8] @DRAM
// )
void gemm_NEON_8x6_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_8x7_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 8] @DRAM
// )
void gemm_NEON_8x7_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_8x7_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][7, 8] @DRAM
// )
void gemm_NEON_8x7_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float *C = (float*) malloc(8 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
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

// gemm_NEON_8x8_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_NEON_8x8_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + (4) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(4) * (ldc) + (4) * (1)], C_reg_4_1);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(5) * (ldc) + (4) * (1)], C_reg_5_1);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(6) * (ldc) + (4) * (1)], C_reg_6_1);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(7) * (ldc) + (4) * (1)], C_reg_7_1);
}

// gemm_NEON_8x8_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 8] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][8, 8] @DRAM
// )
void gemm_NEON_8x8_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_0_1 = vld1q_f32(&C[(4) * (1)]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_1_1 = vld1q_f32(&C[ldc + (4) * (1)]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_2_1 = vld1q_f32(&C[(2) * (ldc) + (4) * (1)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_3_1 = vld1q_f32(&C[(3) * (ldc) + (4) * (1)]);
C_reg_4_0 = vld1q_f32(&C[(4) * (ldc)]);
C_reg_4_1 = vld1q_f32(&C[(4) * (ldc) + (4) * (1)]);
C_reg_5_0 = vld1q_f32(&C[(5) * (ldc)]);
C_reg_5_1 = vld1q_f32(&C[(5) * (ldc) + (4) * (1)]);
C_reg_6_0 = vld1q_f32(&C[(6) * (ldc)]);
C_reg_6_1 = vld1q_f32(&C[(6) * (ldc) + (4) * (1)]);
C_reg_7_0 = vld1q_f32(&C[(7) * (ldc)]);
C_reg_7_1 = vld1q_f32(&C[(7) * (ldc) + (4) * (1)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + (4) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + (4) * (1)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
vst1q_f32(&C[(4) * (ldc)], C_reg_4_0);
vst1q_f32(&C[(4) * (ldc) + (4) * (1)], C_reg_4_1);
vst1q_f32(&C[(5) * (ldc)], C_reg_5_0);
vst1q_f32(&C[(5) * (ldc) + (4) * (1)], C_reg_5_1);
vst1q_f32(&C[(6) * (ldc)], C_reg_6_0);
vst1q_f32(&C[(6) * (ldc) + (4) * (1)], C_reg_6_1);
vst1q_f32(&C[(7) * (ldc)], C_reg_7_0);
vst1q_f32(&C[(7) * (ldc) + (4) * (1)], C_reg_7_1);
}

// gemm_NEON_8x9_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 8] @DRAM
// )
void gemm_NEON_8x9_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[7 * ldci] = C[56];
Ci[7 * ldci + 1] = C[57];
Ci[7 * ldci + 2] = C[58];
Ci[7 * ldci + 3] = C[59];
Ci[7 * ldci + 4] = C[60];
Ci[7 * ldci + 5] = C[61];
Ci[7 * ldci + 6] = C[62];
Ci[7 * ldci + 7] = C[63];
Ci[8 * ldci] = C[64];
Ci[8 * ldci + 1] = C[65];
Ci[8 * ldci + 2] = C[66];
Ci[8 * ldci + 3] = C[67];
Ci[8 * ldci + 4] = C[68];
Ci[8 * ldci + 5] = C[69];
Ci[8 * ldci + 6] = C[70];
Ci[8 * ldci + 7] = C[71];
free(C);
}

// gemm_NEON_8x9_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 8] @DRAM,
//     B : [f32][KC, 12] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][9, 8] @DRAM
// )
void gemm_NEON_8x9_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t B_reg_0, B_reg_1, B_reg_2;;
float *C = (float*) malloc(12 * 8 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_4_0;
float32x4_t C_reg_4_1;
float32x4_t C_reg_5_0;
float32x4_t C_reg_5_1;
float32x4_t C_reg_6_0;
float32x4_t C_reg_6_1;
float32x4_t C_reg_7_0;
float32x4_t C_reg_7_1;
float32x4_t C_reg_8_0;
float32x4_t C_reg_8_1;
float32x4_t C_reg_9_0;
float32x4_t C_reg_9_1;
float32x4_t C_reg_10_0;
float32x4_t C_reg_10_1;
float32x4_t C_reg_11_0;
float32x4_t C_reg_11_1;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_4_0 = vmovq_n_f32(0.0f);
C_reg_4_1 = vmovq_n_f32(0.0f);
C_reg_5_0 = vmovq_n_f32(0.0f);
C_reg_5_1 = vmovq_n_f32(0.0f);
C_reg_6_0 = vmovq_n_f32(0.0f);
C_reg_6_1 = vmovq_n_f32(0.0f);
C_reg_7_0 = vmovq_n_f32(0.0f);
C_reg_7_1 = vmovq_n_f32(0.0f);
C_reg_8_0 = vmovq_n_f32(0.0f);
C_reg_8_1 = vmovq_n_f32(0.0f);
C_reg_9_0 = vmovq_n_f32(0.0f);
C_reg_9_1 = vmovq_n_f32(0.0f);
C_reg_10_0 = vmovq_n_f32(0.0f);
C_reg_10_1 = vmovq_n_f32(0.0f);
C_reg_11_0 = vmovq_n_f32(0.0f);
C_reg_11_1 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (8)]);
  A_reg_1 = vld1q_f32(&A[(k) * (8) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (20)]);
  B_reg_1 = vld1q_f32(&B[(k) * (20) + 4]);
  B_reg_2 = vld1q_f32(&B[(k) * (20) + 8]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_4_0 = vfmaq_laneq_f32(C_reg_4_0, A_reg_0, B_reg_1, (0));
  C_reg_5_0 = vfmaq_laneq_f32(C_reg_5_0, A_reg_0, B_reg_1, (1));
  C_reg_6_0 = vfmaq_laneq_f32(C_reg_6_0, A_reg_0, B_reg_1, (2));
  C_reg_7_0 = vfmaq_laneq_f32(C_reg_7_0, A_reg_0, B_reg_1, (3));
  C_reg_4_1 = vfmaq_laneq_f32(C_reg_4_1, A_reg_1, B_reg_1, (0));
  C_reg_5_1 = vfmaq_laneq_f32(C_reg_5_1, A_reg_1, B_reg_1, (1));
  C_reg_6_1 = vfmaq_laneq_f32(C_reg_6_1, A_reg_1, B_reg_1, (2));
  C_reg_7_1 = vfmaq_laneq_f32(C_reg_7_1, A_reg_1, B_reg_1, (3));
  C_reg_8_0 = vfmaq_laneq_f32(C_reg_8_0, A_reg_0, B_reg_2, (0));
  C_reg_9_0 = vfmaq_laneq_f32(C_reg_9_0, A_reg_0, B_reg_2, (1));
  C_reg_10_0 = vfmaq_laneq_f32(C_reg_10_0, A_reg_0, B_reg_2, (2));
  C_reg_11_0 = vfmaq_laneq_f32(C_reg_11_0, A_reg_0, B_reg_2, (3));
  C_reg_8_1 = vfmaq_laneq_f32(C_reg_8_1, A_reg_1, B_reg_2, (0));
  C_reg_9_1 = vfmaq_laneq_f32(C_reg_9_1, A_reg_1, B_reg_2, (1));
  C_reg_10_1 = vfmaq_laneq_f32(C_reg_10_1, A_reg_1, B_reg_2, (2));
  C_reg_11_1 = vfmaq_laneq_f32(C_reg_11_1, A_reg_1, B_reg_2, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_1_0);
vst1q_f32(&C[8 + 4], C_reg_1_1);
vst1q_f32(&C[(2) * 8], C_reg_2_0);
vst1q_f32(&C[(2) * 8 + 4], C_reg_2_1);
vst1q_f32(&C[(3) * 8], C_reg_3_0);
vst1q_f32(&C[(3) * 8 + 4], C_reg_3_1);
vst1q_f32(&C[(4) * 8], C_reg_4_0);
vst1q_f32(&C[(4) * 8 + 4], C_reg_4_1);
vst1q_f32(&C[(5) * 8], C_reg_5_0);
vst1q_f32(&C[(5) * 8 + 4], C_reg_5_1);
vst1q_f32(&C[(6) * 8], C_reg_6_0);
vst1q_f32(&C[(6) * 8 + 4], C_reg_6_1);
vst1q_f32(&C[(7) * 8], C_reg_7_0);
vst1q_f32(&C[(7) * 8 + 4], C_reg_7_1);
vst1q_f32(&C[(8) * 8], C_reg_8_0);
vst1q_f32(&C[(8) * 8 + 4], C_reg_8_1);
vst1q_f32(&C[(9) * 8], C_reg_9_0);
vst1q_f32(&C[(9) * 8 + 4], C_reg_9_1);
vst1q_f32(&C[(10) * 8], C_reg_10_0);
vst1q_f32(&C[(10) * 8 + 4], C_reg_10_1);
vst1q_f32(&C[(11) * 8], C_reg_11_0);
vst1q_f32(&C[(11) * 8 + 4], C_reg_11_1);
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
Ci[7 * ldci] += C[56];
Ci[7 * ldci + 1] += C[57];
Ci[7 * ldci + 2] += C[58];
Ci[7 * ldci + 3] += C[59];
Ci[7 * ldci + 4] += C[60];
Ci[7 * ldci + 5] += C[61];
Ci[7 * ldci + 6] += C[62];
Ci[7 * ldci + 7] += C[63];
Ci[8 * ldci] += C[64];
Ci[8 * ldci + 1] += C[65];
Ci[8 * ldci + 2] += C[66];
Ci[8 * ldci + 3] += C[67];
Ci[8 * ldci + 4] += C[68];
Ci[8 * ldci + 5] += C[69];
Ci[8 * ldci + 6] += C[70];
Ci[8 * ldci + 7] += C[71];
free(C);
}


/* relying on the following instruction..."
neon_vfmla_4xf32_4xf32(dst,lhs,rhs,jtt)
{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {jtt});
*/

/* relying on the following instruction..."
neon_vfmla_4xf32_4xf32_ori(dst,lhs,rhs,jtt)
{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {jtt});
*/

/* relying on the following instruction..."
neon_vld_4xf32(dst,src,e)
{dst_data} = vld1q_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vst_4xf32(dst,src,e)
vst1q_f32(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
neon_zero_4xf32(dst)
{dst_data} = vmovq_n_f32(0.0f);
*/
