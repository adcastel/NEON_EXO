#include "kernels_NEON_16x4_fp32.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


// gemm_NEON_10x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 10] @DRAM
// )
void gemm_NEON_10x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
free(C);
}

// gemm_NEON_10x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 10] @DRAM
// )
void gemm_NEON_10x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
free(C);
}

// gemm_NEON_10x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 10] @DRAM
// )
void gemm_NEON_10x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[ldci + 9] = C[21];
free(C);
}

// gemm_NEON_10x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 10] @DRAM
// )
void gemm_NEON_10x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[ldci + 9] += C[21];
free(C);
}

// gemm_NEON_10x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 10] @DRAM
// )
void gemm_NEON_10x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[ldci + 9] = C[21];
Ci[2 * ldci] = C[24];
Ci[2 * ldci + 1] = C[25];
Ci[2 * ldci + 2] = C[26];
Ci[2 * ldci + 3] = C[27];
Ci[2 * ldci + 4] = C[28];
Ci[2 * ldci + 5] = C[29];
Ci[2 * ldci + 6] = C[30];
Ci[2 * ldci + 7] = C[31];
Ci[2 * ldci + 8] = C[32];
Ci[2 * ldci + 9] = C[33];
free(C);
}

// gemm_NEON_10x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 10] @DRAM
// )
void gemm_NEON_10x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[ldci + 9] += C[21];
Ci[2 * ldci] += C[24];
Ci[2 * ldci + 1] += C[25];
Ci[2 * ldci + 2] += C[26];
Ci[2 * ldci + 3] += C[27];
Ci[2 * ldci + 4] += C[28];
Ci[2 * ldci + 5] += C[29];
Ci[2 * ldci + 6] += C[30];
Ci[2 * ldci + 7] += C[31];
Ci[2 * ldci + 8] += C[32];
Ci[2 * ldci + 9] += C[33];
free(C);
}

// gemm_NEON_10x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 10] @DRAM
// )
void gemm_NEON_10x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[ldci + 9] = C[21];
Ci[2 * ldci] = C[24];
Ci[2 * ldci + 1] = C[25];
Ci[2 * ldci + 2] = C[26];
Ci[2 * ldci + 3] = C[27];
Ci[2 * ldci + 4] = C[28];
Ci[2 * ldci + 5] = C[29];
Ci[2 * ldci + 6] = C[30];
Ci[2 * ldci + 7] = C[31];
Ci[2 * ldci + 8] = C[32];
Ci[2 * ldci + 9] = C[33];
Ci[3 * ldci] = C[36];
Ci[3 * ldci + 1] = C[37];
Ci[3 * ldci + 2] = C[38];
Ci[3 * ldci + 3] = C[39];
Ci[3 * ldci + 4] = C[40];
Ci[3 * ldci + 5] = C[41];
Ci[3 * ldci + 6] = C[42];
Ci[3 * ldci + 7] = C[43];
Ci[3 * ldci + 8] = C[44];
Ci[3 * ldci + 9] = C[45];
free(C);
}

// gemm_NEON_10x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 10] @DRAM
// )
void gemm_NEON_10x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[ldci + 9] += C[21];
Ci[2 * ldci] += C[24];
Ci[2 * ldci + 1] += C[25];
Ci[2 * ldci + 2] += C[26];
Ci[2 * ldci + 3] += C[27];
Ci[2 * ldci + 4] += C[28];
Ci[2 * ldci + 5] += C[29];
Ci[2 * ldci + 6] += C[30];
Ci[2 * ldci + 7] += C[31];
Ci[2 * ldci + 8] += C[32];
Ci[2 * ldci + 9] += C[33];
Ci[3 * ldci] += C[36];
Ci[3 * ldci + 1] += C[37];
Ci[3 * ldci + 2] += C[38];
Ci[3 * ldci + 3] += C[39];
Ci[3 * ldci + 4] += C[40];
Ci[3 * ldci + 5] += C[41];
Ci[3 * ldci + 6] += C[42];
Ci[3 * ldci + 7] += C[43];
Ci[3 * ldci + 8] += C[44];
Ci[3 * ldci + 9] += C[45];
free(C);
}

// gemm_NEON_11x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 11] @DRAM
// )
void gemm_NEON_11x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
free(C);
}

// gemm_NEON_11x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 11] @DRAM
// )
void gemm_NEON_11x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
free(C);
}

// gemm_NEON_11x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 11] @DRAM
// )
void gemm_NEON_11x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[ldci + 9] = C[21];
Ci[ldci + 10] = C[22];
free(C);
}

// gemm_NEON_11x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 11] @DRAM
// )
void gemm_NEON_11x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[ldci + 9] += C[21];
Ci[ldci + 10] += C[22];
free(C);
}

// gemm_NEON_11x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 11] @DRAM
// )
void gemm_NEON_11x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[ldci + 9] = C[21];
Ci[ldci + 10] = C[22];
Ci[2 * ldci] = C[24];
Ci[2 * ldci + 1] = C[25];
Ci[2 * ldci + 2] = C[26];
Ci[2 * ldci + 3] = C[27];
Ci[2 * ldci + 4] = C[28];
Ci[2 * ldci + 5] = C[29];
Ci[2 * ldci + 6] = C[30];
Ci[2 * ldci + 7] = C[31];
Ci[2 * ldci + 8] = C[32];
Ci[2 * ldci + 9] = C[33];
Ci[2 * ldci + 10] = C[34];
free(C);
}

// gemm_NEON_11x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 11] @DRAM
// )
void gemm_NEON_11x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[ldci + 9] += C[21];
Ci[ldci + 10] += C[22];
Ci[2 * ldci] += C[24];
Ci[2 * ldci + 1] += C[25];
Ci[2 * ldci + 2] += C[26];
Ci[2 * ldci + 3] += C[27];
Ci[2 * ldci + 4] += C[28];
Ci[2 * ldci + 5] += C[29];
Ci[2 * ldci + 6] += C[30];
Ci[2 * ldci + 7] += C[31];
Ci[2 * ldci + 8] += C[32];
Ci[2 * ldci + 9] += C[33];
Ci[2 * ldci + 10] += C[34];
free(C);
}

// gemm_NEON_11x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 11] @DRAM
// )
void gemm_NEON_11x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[ldci + 9] = C[21];
Ci[ldci + 10] = C[22];
Ci[2 * ldci] = C[24];
Ci[2 * ldci + 1] = C[25];
Ci[2 * ldci + 2] = C[26];
Ci[2 * ldci + 3] = C[27];
Ci[2 * ldci + 4] = C[28];
Ci[2 * ldci + 5] = C[29];
Ci[2 * ldci + 6] = C[30];
Ci[2 * ldci + 7] = C[31];
Ci[2 * ldci + 8] = C[32];
Ci[2 * ldci + 9] = C[33];
Ci[2 * ldci + 10] = C[34];
Ci[3 * ldci] = C[36];
Ci[3 * ldci + 1] = C[37];
Ci[3 * ldci + 2] = C[38];
Ci[3 * ldci + 3] = C[39];
Ci[3 * ldci + 4] = C[40];
Ci[3 * ldci + 5] = C[41];
Ci[3 * ldci + 6] = C[42];
Ci[3 * ldci + 7] = C[43];
Ci[3 * ldci + 8] = C[44];
Ci[3 * ldci + 9] = C[45];
Ci[3 * ldci + 10] = C[46];
free(C);
}

// gemm_NEON_11x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 11] @DRAM
// )
void gemm_NEON_11x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[ldci + 9] += C[21];
Ci[ldci + 10] += C[22];
Ci[2 * ldci] += C[24];
Ci[2 * ldci + 1] += C[25];
Ci[2 * ldci + 2] += C[26];
Ci[2 * ldci + 3] += C[27];
Ci[2 * ldci + 4] += C[28];
Ci[2 * ldci + 5] += C[29];
Ci[2 * ldci + 6] += C[30];
Ci[2 * ldci + 7] += C[31];
Ci[2 * ldci + 8] += C[32];
Ci[2 * ldci + 9] += C[33];
Ci[2 * ldci + 10] += C[34];
Ci[3 * ldci] += C[36];
Ci[3 * ldci + 1] += C[37];
Ci[3 * ldci + 2] += C[38];
Ci[3 * ldci + 3] += C[39];
Ci[3 * ldci + 4] += C[40];
Ci[3 * ldci + 5] += C[41];
Ci[3 * ldci + 6] += C[42];
Ci[3 * ldci + 7] += C[43];
Ci[3 * ldci + 8] += C[44];
Ci[3 * ldci + 9] += C[45];
Ci[3 * ldci + 10] += C[46];
free(C);
}

// gemm_NEON_12x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 12] @DRAM
// )
void gemm_NEON_12x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
free(C);
}

// gemm_NEON_12x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 12] @DRAM
// )
void gemm_NEON_12x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
free(C);
}

// gemm_NEON_12x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 12] @DRAM
// )
void gemm_NEON_12x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[ldci + 9] = C[21];
Ci[ldci + 10] = C[22];
Ci[ldci + 11] = C[23];
free(C);
}

// gemm_NEON_12x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 12] @DRAM
// )
void gemm_NEON_12x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[ldci + 9] += C[21];
Ci[ldci + 10] += C[22];
Ci[ldci + 11] += C[23];
free(C);
}

// gemm_NEON_12x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 12] @DRAM
// )
void gemm_NEON_12x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[ldci + 9] = C[21];
Ci[ldci + 10] = C[22];
Ci[ldci + 11] = C[23];
Ci[2 * ldci] = C[24];
Ci[2 * ldci + 1] = C[25];
Ci[2 * ldci + 2] = C[26];
Ci[2 * ldci + 3] = C[27];
Ci[2 * ldci + 4] = C[28];
Ci[2 * ldci + 5] = C[29];
Ci[2 * ldci + 6] = C[30];
Ci[2 * ldci + 7] = C[31];
Ci[2 * ldci + 8] = C[32];
Ci[2 * ldci + 9] = C[33];
Ci[2 * ldci + 10] = C[34];
Ci[2 * ldci + 11] = C[35];
free(C);
}

// gemm_NEON_12x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 12] @DRAM
// )
void gemm_NEON_12x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[ldci + 9] += C[21];
Ci[ldci + 10] += C[22];
Ci[ldci + 11] += C[23];
Ci[2 * ldci] += C[24];
Ci[2 * ldci + 1] += C[25];
Ci[2 * ldci + 2] += C[26];
Ci[2 * ldci + 3] += C[27];
Ci[2 * ldci + 4] += C[28];
Ci[2 * ldci + 5] += C[29];
Ci[2 * ldci + 6] += C[30];
Ci[2 * ldci + 7] += C[31];
Ci[2 * ldci + 8] += C[32];
Ci[2 * ldci + 9] += C[33];
Ci[2 * ldci + 10] += C[34];
Ci[2 * ldci + 11] += C[35];
free(C);
}

// gemm_NEON_12x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_NEON_12x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + (4) * (1)]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + (8) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[(8) * (1)], C_reg_0_2);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[ldc + (8) * (1)], C_reg_1_2);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(2) * (ldc) + (8) * (1)], C_reg_2_2);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
vst1q_f32(&C[(3) * (ldc) + (8) * (1)], C_reg_3_2);
}

// gemm_NEON_12x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 12] @DRAM
// )
void gemm_NEON_12x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_0_1 = vld1q_f32(&C[(4) * (1)]);
C_reg_0_2 = vld1q_f32(&C[(8) * (1)]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_1_1 = vld1q_f32(&C[ldc + (4) * (1)]);
C_reg_1_2 = vld1q_f32(&C[ldc + (8) * (1)]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_2_1 = vld1q_f32(&C[(2) * (ldc) + (4) * (1)]);
C_reg_2_2 = vld1q_f32(&C[(2) * (ldc) + (8) * (1)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_3_1 = vld1q_f32(&C[(3) * (ldc) + (4) * (1)]);
C_reg_3_2 = vld1q_f32(&C[(3) * (ldc) + (8) * (1)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + (4) * (1)]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + (8) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[(8) * (1)], C_reg_0_2);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[ldc + (8) * (1)], C_reg_1_2);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(2) * (ldc) + (8) * (1)], C_reg_2_2);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
vst1q_f32(&C[(3) * (ldc) + (8) * (1)], C_reg_3_2);
}

// gemm_NEON_13x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 13] @DRAM
// )
void gemm_NEON_13x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
free(C);
}

// gemm_NEON_13x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 13] @DRAM
// )
void gemm_NEON_13x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
free(C);
}

// gemm_NEON_13x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 13] @DRAM
// )
void gemm_NEON_13x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
free(C);
}

// gemm_NEON_13x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 13] @DRAM
// )
void gemm_NEON_13x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
free(C);
}

// gemm_NEON_13x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 13] @DRAM
// )
void gemm_NEON_13x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[2 * ldci] = C[32];
Ci[2 * ldci + 1] = C[33];
Ci[2 * ldci + 2] = C[34];
Ci[2 * ldci + 3] = C[35];
Ci[2 * ldci + 4] = C[36];
Ci[2 * ldci + 5] = C[37];
Ci[2 * ldci + 6] = C[38];
Ci[2 * ldci + 7] = C[39];
Ci[2 * ldci + 8] = C[40];
Ci[2 * ldci + 9] = C[41];
Ci[2 * ldci + 10] = C[42];
Ci[2 * ldci + 11] = C[43];
Ci[2 * ldci + 12] = C[44];
free(C);
}

// gemm_NEON_13x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 13] @DRAM
// )
void gemm_NEON_13x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[2 * ldci] += C[32];
Ci[2 * ldci + 1] += C[33];
Ci[2 * ldci + 2] += C[34];
Ci[2 * ldci + 3] += C[35];
Ci[2 * ldci + 4] += C[36];
Ci[2 * ldci + 5] += C[37];
Ci[2 * ldci + 6] += C[38];
Ci[2 * ldci + 7] += C[39];
Ci[2 * ldci + 8] += C[40];
Ci[2 * ldci + 9] += C[41];
Ci[2 * ldci + 10] += C[42];
Ci[2 * ldci + 11] += C[43];
Ci[2 * ldci + 12] += C[44];
free(C);
}

// gemm_NEON_13x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 13] @DRAM
// )
void gemm_NEON_13x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[2 * ldci] = C[32];
Ci[2 * ldci + 1] = C[33];
Ci[2 * ldci + 2] = C[34];
Ci[2 * ldci + 3] = C[35];
Ci[2 * ldci + 4] = C[36];
Ci[2 * ldci + 5] = C[37];
Ci[2 * ldci + 6] = C[38];
Ci[2 * ldci + 7] = C[39];
Ci[2 * ldci + 8] = C[40];
Ci[2 * ldci + 9] = C[41];
Ci[2 * ldci + 10] = C[42];
Ci[2 * ldci + 11] = C[43];
Ci[2 * ldci + 12] = C[44];
Ci[3 * ldci] = C[48];
Ci[3 * ldci + 1] = C[49];
Ci[3 * ldci + 2] = C[50];
Ci[3 * ldci + 3] = C[51];
Ci[3 * ldci + 4] = C[52];
Ci[3 * ldci + 5] = C[53];
Ci[3 * ldci + 6] = C[54];
Ci[3 * ldci + 7] = C[55];
Ci[3 * ldci + 8] = C[56];
Ci[3 * ldci + 9] = C[57];
Ci[3 * ldci + 10] = C[58];
Ci[3 * ldci + 11] = C[59];
Ci[3 * ldci + 12] = C[60];
free(C);
}

// gemm_NEON_13x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 13] @DRAM
// )
void gemm_NEON_13x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[2 * ldci] += C[32];
Ci[2 * ldci + 1] += C[33];
Ci[2 * ldci + 2] += C[34];
Ci[2 * ldci + 3] += C[35];
Ci[2 * ldci + 4] += C[36];
Ci[2 * ldci + 5] += C[37];
Ci[2 * ldci + 6] += C[38];
Ci[2 * ldci + 7] += C[39];
Ci[2 * ldci + 8] += C[40];
Ci[2 * ldci + 9] += C[41];
Ci[2 * ldci + 10] += C[42];
Ci[2 * ldci + 11] += C[43];
Ci[2 * ldci + 12] += C[44];
Ci[3 * ldci] += C[48];
Ci[3 * ldci + 1] += C[49];
Ci[3 * ldci + 2] += C[50];
Ci[3 * ldci + 3] += C[51];
Ci[3 * ldci + 4] += C[52];
Ci[3 * ldci + 5] += C[53];
Ci[3 * ldci + 6] += C[54];
Ci[3 * ldci + 7] += C[55];
Ci[3 * ldci + 8] += C[56];
Ci[3 * ldci + 9] += C[57];
Ci[3 * ldci + 10] += C[58];
Ci[3 * ldci + 11] += C[59];
Ci[3 * ldci + 12] += C[60];
free(C);
}

// gemm_NEON_14x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 14] @DRAM
// )
void gemm_NEON_14x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
free(C);
}

// gemm_NEON_14x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 14] @DRAM
// )
void gemm_NEON_14x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
free(C);
}

// gemm_NEON_14x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 14] @DRAM
// )
void gemm_NEON_14x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[ldci + 13] = C[29];
free(C);
}

// gemm_NEON_14x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 14] @DRAM
// )
void gemm_NEON_14x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[ldci + 13] += C[29];
free(C);
}

// gemm_NEON_14x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 14] @DRAM
// )
void gemm_NEON_14x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[ldci + 13] = C[29];
Ci[2 * ldci] = C[32];
Ci[2 * ldci + 1] = C[33];
Ci[2 * ldci + 2] = C[34];
Ci[2 * ldci + 3] = C[35];
Ci[2 * ldci + 4] = C[36];
Ci[2 * ldci + 5] = C[37];
Ci[2 * ldci + 6] = C[38];
Ci[2 * ldci + 7] = C[39];
Ci[2 * ldci + 8] = C[40];
Ci[2 * ldci + 9] = C[41];
Ci[2 * ldci + 10] = C[42];
Ci[2 * ldci + 11] = C[43];
Ci[2 * ldci + 12] = C[44];
Ci[2 * ldci + 13] = C[45];
free(C);
}

// gemm_NEON_14x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 14] @DRAM
// )
void gemm_NEON_14x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[ldci + 13] += C[29];
Ci[2 * ldci] += C[32];
Ci[2 * ldci + 1] += C[33];
Ci[2 * ldci + 2] += C[34];
Ci[2 * ldci + 3] += C[35];
Ci[2 * ldci + 4] += C[36];
Ci[2 * ldci + 5] += C[37];
Ci[2 * ldci + 6] += C[38];
Ci[2 * ldci + 7] += C[39];
Ci[2 * ldci + 8] += C[40];
Ci[2 * ldci + 9] += C[41];
Ci[2 * ldci + 10] += C[42];
Ci[2 * ldci + 11] += C[43];
Ci[2 * ldci + 12] += C[44];
Ci[2 * ldci + 13] += C[45];
free(C);
}

// gemm_NEON_14x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 14] @DRAM
// )
void gemm_NEON_14x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[ldci + 13] = C[29];
Ci[2 * ldci] = C[32];
Ci[2 * ldci + 1] = C[33];
Ci[2 * ldci + 2] = C[34];
Ci[2 * ldci + 3] = C[35];
Ci[2 * ldci + 4] = C[36];
Ci[2 * ldci + 5] = C[37];
Ci[2 * ldci + 6] = C[38];
Ci[2 * ldci + 7] = C[39];
Ci[2 * ldci + 8] = C[40];
Ci[2 * ldci + 9] = C[41];
Ci[2 * ldci + 10] = C[42];
Ci[2 * ldci + 11] = C[43];
Ci[2 * ldci + 12] = C[44];
Ci[2 * ldci + 13] = C[45];
Ci[3 * ldci] = C[48];
Ci[3 * ldci + 1] = C[49];
Ci[3 * ldci + 2] = C[50];
Ci[3 * ldci + 3] = C[51];
Ci[3 * ldci + 4] = C[52];
Ci[3 * ldci + 5] = C[53];
Ci[3 * ldci + 6] = C[54];
Ci[3 * ldci + 7] = C[55];
Ci[3 * ldci + 8] = C[56];
Ci[3 * ldci + 9] = C[57];
Ci[3 * ldci + 10] = C[58];
Ci[3 * ldci + 11] = C[59];
Ci[3 * ldci + 12] = C[60];
Ci[3 * ldci + 13] = C[61];
free(C);
}

// gemm_NEON_14x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 14] @DRAM
// )
void gemm_NEON_14x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[ldci + 13] += C[29];
Ci[2 * ldci] += C[32];
Ci[2 * ldci + 1] += C[33];
Ci[2 * ldci + 2] += C[34];
Ci[2 * ldci + 3] += C[35];
Ci[2 * ldci + 4] += C[36];
Ci[2 * ldci + 5] += C[37];
Ci[2 * ldci + 6] += C[38];
Ci[2 * ldci + 7] += C[39];
Ci[2 * ldci + 8] += C[40];
Ci[2 * ldci + 9] += C[41];
Ci[2 * ldci + 10] += C[42];
Ci[2 * ldci + 11] += C[43];
Ci[2 * ldci + 12] += C[44];
Ci[2 * ldci + 13] += C[45];
Ci[3 * ldci] += C[48];
Ci[3 * ldci + 1] += C[49];
Ci[3 * ldci + 2] += C[50];
Ci[3 * ldci + 3] += C[51];
Ci[3 * ldci + 4] += C[52];
Ci[3 * ldci + 5] += C[53];
Ci[3 * ldci + 6] += C[54];
Ci[3 * ldci + 7] += C[55];
Ci[3 * ldci + 8] += C[56];
Ci[3 * ldci + 9] += C[57];
Ci[3 * ldci + 10] += C[58];
Ci[3 * ldci + 11] += C[59];
Ci[3 * ldci + 12] += C[60];
Ci[3 * ldci + 13] += C[61];
free(C);
}

// gemm_NEON_15x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 15] @DRAM
// )
void gemm_NEON_15x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[14] = C[14];
free(C);
}

// gemm_NEON_15x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 15] @DRAM
// )
void gemm_NEON_15x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[14] += C[14];
free(C);
}

// gemm_NEON_15x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 15] @DRAM
// )
void gemm_NEON_15x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[14] = C[14];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[ldci + 13] = C[29];
Ci[ldci + 14] = C[30];
free(C);
}

// gemm_NEON_15x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 15] @DRAM
// )
void gemm_NEON_15x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[14] += C[14];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[ldci + 13] += C[29];
Ci[ldci + 14] += C[30];
free(C);
}

// gemm_NEON_15x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 15] @DRAM
// )
void gemm_NEON_15x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[14] = C[14];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[ldci + 13] = C[29];
Ci[ldci + 14] = C[30];
Ci[2 * ldci] = C[32];
Ci[2 * ldci + 1] = C[33];
Ci[2 * ldci + 2] = C[34];
Ci[2 * ldci + 3] = C[35];
Ci[2 * ldci + 4] = C[36];
Ci[2 * ldci + 5] = C[37];
Ci[2 * ldci + 6] = C[38];
Ci[2 * ldci + 7] = C[39];
Ci[2 * ldci + 8] = C[40];
Ci[2 * ldci + 9] = C[41];
Ci[2 * ldci + 10] = C[42];
Ci[2 * ldci + 11] = C[43];
Ci[2 * ldci + 12] = C[44];
Ci[2 * ldci + 13] = C[45];
Ci[2 * ldci + 14] = C[46];
free(C);
}

// gemm_NEON_15x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 15] @DRAM
// )
void gemm_NEON_15x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[14] += C[14];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[ldci + 13] += C[29];
Ci[ldci + 14] += C[30];
Ci[2 * ldci] += C[32];
Ci[2 * ldci + 1] += C[33];
Ci[2 * ldci + 2] += C[34];
Ci[2 * ldci + 3] += C[35];
Ci[2 * ldci + 4] += C[36];
Ci[2 * ldci + 5] += C[37];
Ci[2 * ldci + 6] += C[38];
Ci[2 * ldci + 7] += C[39];
Ci[2 * ldci + 8] += C[40];
Ci[2 * ldci + 9] += C[41];
Ci[2 * ldci + 10] += C[42];
Ci[2 * ldci + 11] += C[43];
Ci[2 * ldci + 12] += C[44];
Ci[2 * ldci + 13] += C[45];
Ci[2 * ldci + 14] += C[46];
free(C);
}

// gemm_NEON_15x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 15] @DRAM
// )
void gemm_NEON_15x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[14] = C[14];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[ldci + 13] = C[29];
Ci[ldci + 14] = C[30];
Ci[2 * ldci] = C[32];
Ci[2 * ldci + 1] = C[33];
Ci[2 * ldci + 2] = C[34];
Ci[2 * ldci + 3] = C[35];
Ci[2 * ldci + 4] = C[36];
Ci[2 * ldci + 5] = C[37];
Ci[2 * ldci + 6] = C[38];
Ci[2 * ldci + 7] = C[39];
Ci[2 * ldci + 8] = C[40];
Ci[2 * ldci + 9] = C[41];
Ci[2 * ldci + 10] = C[42];
Ci[2 * ldci + 11] = C[43];
Ci[2 * ldci + 12] = C[44];
Ci[2 * ldci + 13] = C[45];
Ci[2 * ldci + 14] = C[46];
Ci[3 * ldci] = C[48];
Ci[3 * ldci + 1] = C[49];
Ci[3 * ldci + 2] = C[50];
Ci[3 * ldci + 3] = C[51];
Ci[3 * ldci + 4] = C[52];
Ci[3 * ldci + 5] = C[53];
Ci[3 * ldci + 6] = C[54];
Ci[3 * ldci + 7] = C[55];
Ci[3 * ldci + 8] = C[56];
Ci[3 * ldci + 9] = C[57];
Ci[3 * ldci + 10] = C[58];
Ci[3 * ldci + 11] = C[59];
Ci[3 * ldci + 12] = C[60];
Ci[3 * ldci + 13] = C[61];
Ci[3 * ldci + 14] = C[62];
free(C);
}

// gemm_NEON_15x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 15] @DRAM
// )
void gemm_NEON_15x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[14] += C[14];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[ldci + 13] += C[29];
Ci[ldci + 14] += C[30];
Ci[2 * ldci] += C[32];
Ci[2 * ldci + 1] += C[33];
Ci[2 * ldci + 2] += C[34];
Ci[2 * ldci + 3] += C[35];
Ci[2 * ldci + 4] += C[36];
Ci[2 * ldci + 5] += C[37];
Ci[2 * ldci + 6] += C[38];
Ci[2 * ldci + 7] += C[39];
Ci[2 * ldci + 8] += C[40];
Ci[2 * ldci + 9] += C[41];
Ci[2 * ldci + 10] += C[42];
Ci[2 * ldci + 11] += C[43];
Ci[2 * ldci + 12] += C[44];
Ci[2 * ldci + 13] += C[45];
Ci[2 * ldci + 14] += C[46];
Ci[3 * ldci] += C[48];
Ci[3 * ldci + 1] += C[49];
Ci[3 * ldci + 2] += C[50];
Ci[3 * ldci + 3] += C[51];
Ci[3 * ldci + 4] += C[52];
Ci[3 * ldci + 5] += C[53];
Ci[3 * ldci + 6] += C[54];
Ci[3 * ldci + 7] += C[55];
Ci[3 * ldci + 8] += C[56];
Ci[3 * ldci + 9] += C[57];
Ci[3 * ldci + 10] += C[58];
Ci[3 * ldci + 11] += C[59];
Ci[3 * ldci + 12] += C[60];
Ci[3 * ldci + 13] += C[61];
Ci[3 * ldci + 14] += C[62];
free(C);
}

// gemm_NEON_16x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 16] @DRAM
// )
void gemm_NEON_16x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[14] = C[14];
Ci[15] = C[15];
free(C);
}

// gemm_NEON_16x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 16] @DRAM
// )
void gemm_NEON_16x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[14] += C[14];
Ci[15] += C[15];
free(C);
}

// gemm_NEON_16x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 16] @DRAM
// )
void gemm_NEON_16x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[14] = C[14];
Ci[15] = C[15];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[ldci + 13] = C[29];
Ci[ldci + 14] = C[30];
Ci[ldci + 15] = C[31];
free(C);
}

// gemm_NEON_16x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 16] @DRAM
// )
void gemm_NEON_16x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[14] += C[14];
Ci[15] += C[15];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[ldci + 13] += C[29];
Ci[ldci + 14] += C[30];
Ci[ldci + 15] += C[31];
free(C);
}

// gemm_NEON_16x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 16] @DRAM
// )
void gemm_NEON_16x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[9] = C[9];
Ci[10] = C[10];
Ci[11] = C[11];
Ci[12] = C[12];
Ci[13] = C[13];
Ci[14] = C[14];
Ci[15] = C[15];
Ci[ldci] = C[16];
Ci[ldci + 1] = C[17];
Ci[ldci + 2] = C[18];
Ci[ldci + 3] = C[19];
Ci[ldci + 4] = C[20];
Ci[ldci + 5] = C[21];
Ci[ldci + 6] = C[22];
Ci[ldci + 7] = C[23];
Ci[ldci + 8] = C[24];
Ci[ldci + 9] = C[25];
Ci[ldci + 10] = C[26];
Ci[ldci + 11] = C[27];
Ci[ldci + 12] = C[28];
Ci[ldci + 13] = C[29];
Ci[ldci + 14] = C[30];
Ci[ldci + 15] = C[31];
Ci[2 * ldci] = C[32];
Ci[2 * ldci + 1] = C[33];
Ci[2 * ldci + 2] = C[34];
Ci[2 * ldci + 3] = C[35];
Ci[2 * ldci + 4] = C[36];
Ci[2 * ldci + 5] = C[37];
Ci[2 * ldci + 6] = C[38];
Ci[2 * ldci + 7] = C[39];
Ci[2 * ldci + 8] = C[40];
Ci[2 * ldci + 9] = C[41];
Ci[2 * ldci + 10] = C[42];
Ci[2 * ldci + 11] = C[43];
Ci[2 * ldci + 12] = C[44];
Ci[2 * ldci + 13] = C[45];
Ci[2 * ldci + 14] = C[46];
Ci[2 * ldci + 15] = C[47];
free(C);
}

// gemm_NEON_16x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 16] @DRAM
// )
void gemm_NEON_16x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 16 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + 12]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_0_3);
vst1q_f32(&C[16], C_reg_1_0);
vst1q_f32(&C[16 + 4], C_reg_1_1);
vst1q_f32(&C[16 + 8], C_reg_1_2);
vst1q_f32(&C[16 + 12], C_reg_1_3);
vst1q_f32(&C[(2) * (16)], C_reg_2_0);
vst1q_f32(&C[(2) * (16) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (16) + 8], C_reg_2_2);
vst1q_f32(&C[(2) * (16) + 12], C_reg_2_3);
vst1q_f32(&C[(3) * (16)], C_reg_3_0);
vst1q_f32(&C[(3) * (16) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (16) + 8], C_reg_3_2);
vst1q_f32(&C[(3) * (16) + 12], C_reg_3_3);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[9] += C[9];
Ci[10] += C[10];
Ci[11] += C[11];
Ci[12] += C[12];
Ci[13] += C[13];
Ci[14] += C[14];
Ci[15] += C[15];
Ci[ldci] += C[16];
Ci[ldci + 1] += C[17];
Ci[ldci + 2] += C[18];
Ci[ldci + 3] += C[19];
Ci[ldci + 4] += C[20];
Ci[ldci + 5] += C[21];
Ci[ldci + 6] += C[22];
Ci[ldci + 7] += C[23];
Ci[ldci + 8] += C[24];
Ci[ldci + 9] += C[25];
Ci[ldci + 10] += C[26];
Ci[ldci + 11] += C[27];
Ci[ldci + 12] += C[28];
Ci[ldci + 13] += C[29];
Ci[ldci + 14] += C[30];
Ci[ldci + 15] += C[31];
Ci[2 * ldci] += C[32];
Ci[2 * ldci + 1] += C[33];
Ci[2 * ldci + 2] += C[34];
Ci[2 * ldci + 3] += C[35];
Ci[2 * ldci + 4] += C[36];
Ci[2 * ldci + 5] += C[37];
Ci[2 * ldci + 6] += C[38];
Ci[2 * ldci + 7] += C[39];
Ci[2 * ldci + 8] += C[40];
Ci[2 * ldci + 9] += C[41];
Ci[2 * ldci + 10] += C[42];
Ci[2 * ldci + 11] += C[43];
Ci[2 * ldci + 12] += C[44];
Ci[2 * ldci + 13] += C[45];
Ci[2 * ldci + 14] += C[46];
Ci[2 * ldci + 15] += C[47];
free(C);
}

// gemm_NEON_16x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_NEON_16x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_0_3 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_1_3 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_2_3 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
C_reg_3_3 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + (4) * (1)]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + (8) * (1)]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + (12) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[(8) * (1)], C_reg_0_2);
vst1q_f32(&C[(12) * (1)], C_reg_0_3);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[ldc + (8) * (1)], C_reg_1_2);
vst1q_f32(&C[ldc + (12) * (1)], C_reg_1_3);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(2) * (ldc) + (8) * (1)], C_reg_2_2);
vst1q_f32(&C[(2) * (ldc) + (12) * (1)], C_reg_2_3);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
vst1q_f32(&C[(3) * (ldc) + (8) * (1)], C_reg_3_2);
vst1q_f32(&C[(3) * (ldc) + (12) * (1)], C_reg_3_3);
}

// gemm_NEON_16x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 16] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     beta : f32[1] @DRAM,
//     C : [f32][4, 16] @DRAM
// )
void gemm_NEON_16x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* beta, float * C, int ldc ) {
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t A_reg_3;
float32x4_t B_reg_0;;
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_0_3;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_1_3;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_2_3;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
float32x4_t C_reg_3_3;
C_reg_0_0 = vld1q_f32(&C[0]);
C_reg_0_1 = vld1q_f32(&C[(4) * (1)]);
C_reg_0_2 = vld1q_f32(&C[(8) * (1)]);
C_reg_0_3 = vld1q_f32(&C[(12) * (1)]);
C_reg_1_0 = vld1q_f32(&C[ldc]);
C_reg_1_1 = vld1q_f32(&C[ldc + (4) * (1)]);
C_reg_1_2 = vld1q_f32(&C[ldc + (8) * (1)]);
C_reg_1_3 = vld1q_f32(&C[ldc + (12) * (1)]);
C_reg_2_0 = vld1q_f32(&C[(2) * (ldc)]);
C_reg_2_1 = vld1q_f32(&C[(2) * (ldc) + (4) * (1)]);
C_reg_2_2 = vld1q_f32(&C[(2) * (ldc) + (8) * (1)]);
C_reg_2_3 = vld1q_f32(&C[(2) * (ldc) + (12) * (1)]);
C_reg_3_0 = vld1q_f32(&C[(3) * (ldc)]);
C_reg_3_1 = vld1q_f32(&C[(3) * (ldc) + (4) * (1)]);
C_reg_3_2 = vld1q_f32(&C[(3) * (ldc) + (8) * (1)]);
C_reg_3_3 = vld1q_f32(&C[(3) * (ldc) + (12) * (1)]);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + (4) * (1)]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + (8) * (1)]);
  A_reg_3 = vld1q_f32(&A[(k) * (16) + (12) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
  C_reg_0_3 = vfmaq_laneq_f32(C_reg_0_3, A_reg_3, B_reg_0, (0));
  C_reg_1_3 = vfmaq_laneq_f32(C_reg_1_3, A_reg_3, B_reg_0, (1));
  C_reg_2_3 = vfmaq_laneq_f32(C_reg_2_3, A_reg_3, B_reg_0, (2));
  C_reg_3_3 = vfmaq_laneq_f32(C_reg_3_3, A_reg_3, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[(4) * (1)], C_reg_0_1);
vst1q_f32(&C[(8) * (1)], C_reg_0_2);
vst1q_f32(&C[(12) * (1)], C_reg_0_3);
vst1q_f32(&C[ldc], C_reg_1_0);
vst1q_f32(&C[ldc + (4) * (1)], C_reg_1_1);
vst1q_f32(&C[ldc + (8) * (1)], C_reg_1_2);
vst1q_f32(&C[ldc + (12) * (1)], C_reg_1_3);
vst1q_f32(&C[(2) * (ldc)], C_reg_2_0);
vst1q_f32(&C[(2) * (ldc) + (4) * (1)], C_reg_2_1);
vst1q_f32(&C[(2) * (ldc) + (8) * (1)], C_reg_2_2);
vst1q_f32(&C[(2) * (ldc) + (12) * (1)], C_reg_2_3);
vst1q_f32(&C[(3) * (ldc)], C_reg_3_0);
vst1q_f32(&C[(3) * (ldc) + (4) * (1)], C_reg_3_1);
vst1q_f32(&C[(3) * (ldc) + (8) * (1)], C_reg_3_2);
vst1q_f32(&C[(3) * (ldc) + (12) * (1)], C_reg_3_3);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] += C[0];
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
  C_reg_0 = vfmaq_laneq_f32(C_reg_0, A_reg, B_reg, (0));
}
vst1q_f32(&C[0], C_reg_0);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
free(C);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (16)]);
  B_reg = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + (4) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + (4) * (1)]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
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

// gemm_NEON_9x1_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 9] @DRAM
// )
void gemm_NEON_9x1_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
free(C);
}

// gemm_NEON_9x1_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][1, 9] @DRAM
// )
void gemm_NEON_9x1_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
free(C);
}

// gemm_NEON_9x2_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 9] @DRAM
// )
void gemm_NEON_9x2_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
free(C);
}

// gemm_NEON_9x2_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][2, 9] @DRAM
// )
void gemm_NEON_9x2_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
free(C);
}

// gemm_NEON_9x3_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 9] @DRAM
// )
void gemm_NEON_9x3_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[2 * ldci] = C[24];
Ci[2 * ldci + 1] = C[25];
Ci[2 * ldci + 2] = C[26];
Ci[2 * ldci + 3] = C[27];
Ci[2 * ldci + 4] = C[28];
Ci[2 * ldci + 5] = C[29];
Ci[2 * ldci + 6] = C[30];
Ci[2 * ldci + 7] = C[31];
Ci[2 * ldci + 8] = C[32];
free(C);
}

// gemm_NEON_9x3_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][3, 9] @DRAM
// )
void gemm_NEON_9x3_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[2 * ldci] += C[24];
Ci[2 * ldci + 1] += C[25];
Ci[2 * ldci + 2] += C[26];
Ci[2 * ldci + 3] += C[27];
Ci[2 * ldci + 4] += C[28];
Ci[2 * ldci + 5] += C[29];
Ci[2 * ldci + 6] += C[30];
Ci[2 * ldci + 7] += C[31];
Ci[2 * ldci + 8] += C[32];
free(C);
}

// gemm_NEON_9x4_b0_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 9] @DRAM
// )
void gemm_NEON_9x4_b0_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] = C[0];
Ci[1] = C[1];
Ci[2] = C[2];
Ci[3] = C[3];
Ci[4] = C[4];
Ci[5] = C[5];
Ci[6] = C[6];
Ci[7] = C[7];
Ci[8] = C[8];
Ci[ldci] = C[12];
Ci[ldci + 1] = C[13];
Ci[ldci + 2] = C[14];
Ci[ldci + 3] = C[15];
Ci[ldci + 4] = C[16];
Ci[ldci + 5] = C[17];
Ci[ldci + 6] = C[18];
Ci[ldci + 7] = C[19];
Ci[ldci + 8] = C[20];
Ci[2 * ldci] = C[24];
Ci[2 * ldci + 1] = C[25];
Ci[2 * ldci + 2] = C[26];
Ci[2 * ldci + 3] = C[27];
Ci[2 * ldci + 4] = C[28];
Ci[2 * ldci + 5] = C[29];
Ci[2 * ldci + 6] = C[30];
Ci[2 * ldci + 7] = C[31];
Ci[2 * ldci + 8] = C[32];
Ci[3 * ldci] = C[36];
Ci[3 * ldci + 1] = C[37];
Ci[3 * ldci + 2] = C[38];
Ci[3 * ldci + 3] = C[39];
Ci[3 * ldci + 4] = C[40];
Ci[3 * ldci + 5] = C[41];
Ci[3 * ldci + 6] = C[42];
Ci[3 * ldci + 7] = C[43];
Ci[3 * ldci + 8] = C[44];
free(C);
}

// gemm_NEON_9x4_b1_col_fp32(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : [f32][KC, 12] @DRAM,
//     B : [f32][KC, 4] @DRAM,
//     b : f32[1] @DRAM,
//     Ci : [f32][4, 9] @DRAM
// )
void gemm_NEON_9x4_b1_col_fp32( void *ctxt, int_fast32_t KC, const float* alpha, float * A, int lda, float * B, int ldb, const float* b, float * Ci, int ldci ) {
// assert stride(A, 1) == 1
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
float32x4_t A_reg_0;
float32x4_t A_reg_1;
float32x4_t A_reg_2;
float32x4_t B_reg_0;;
float *C = (float*) malloc(4 * 12 * sizeof(*C));
float32x4_t C_reg_0_0;
float32x4_t C_reg_0_1;
float32x4_t C_reg_0_2;
float32x4_t C_reg_1_0;
float32x4_t C_reg_1_1;
float32x4_t C_reg_1_2;
float32x4_t C_reg_2_0;
float32x4_t C_reg_2_1;
float32x4_t C_reg_2_2;
float32x4_t C_reg_3_0;
float32x4_t C_reg_3_1;
float32x4_t C_reg_3_2;
C_reg_0_0 = vmovq_n_f32(0.0f);
C_reg_0_1 = vmovq_n_f32(0.0f);
C_reg_0_2 = vmovq_n_f32(0.0f);
C_reg_1_0 = vmovq_n_f32(0.0f);
C_reg_1_1 = vmovq_n_f32(0.0f);
C_reg_1_2 = vmovq_n_f32(0.0f);
C_reg_2_0 = vmovq_n_f32(0.0f);
C_reg_2_1 = vmovq_n_f32(0.0f);
C_reg_2_2 = vmovq_n_f32(0.0f);
C_reg_3_0 = vmovq_n_f32(0.0f);
C_reg_3_1 = vmovq_n_f32(0.0f);
C_reg_3_2 = vmovq_n_f32(0.0f);
for (int_fast32_t k = 0; k < KC; k++) {
  A_reg_0 = vld1q_f32(&A[(k) * (16)]);
  A_reg_1 = vld1q_f32(&A[(k) * (16) + 4]);
  A_reg_2 = vld1q_f32(&A[(k) * (16) + 8]);
  B_reg_0 = vld1q_f32(&B[(k) * (4)]);
  C_reg_0_0 = vfmaq_laneq_f32(C_reg_0_0, A_reg_0, B_reg_0, (0));
  C_reg_1_0 = vfmaq_laneq_f32(C_reg_1_0, A_reg_0, B_reg_0, (1));
  C_reg_2_0 = vfmaq_laneq_f32(C_reg_2_0, A_reg_0, B_reg_0, (2));
  C_reg_3_0 = vfmaq_laneq_f32(C_reg_3_0, A_reg_0, B_reg_0, (3));
  C_reg_0_1 = vfmaq_laneq_f32(C_reg_0_1, A_reg_1, B_reg_0, (0));
  C_reg_1_1 = vfmaq_laneq_f32(C_reg_1_1, A_reg_1, B_reg_0, (1));
  C_reg_2_1 = vfmaq_laneq_f32(C_reg_2_1, A_reg_1, B_reg_0, (2));
  C_reg_3_1 = vfmaq_laneq_f32(C_reg_3_1, A_reg_1, B_reg_0, (3));
  C_reg_0_2 = vfmaq_laneq_f32(C_reg_0_2, A_reg_2, B_reg_0, (0));
  C_reg_1_2 = vfmaq_laneq_f32(C_reg_1_2, A_reg_2, B_reg_0, (1));
  C_reg_2_2 = vfmaq_laneq_f32(C_reg_2_2, A_reg_2, B_reg_0, (2));
  C_reg_3_2 = vfmaq_laneq_f32(C_reg_3_2, A_reg_2, B_reg_0, (3));
}
vst1q_f32(&C[0], C_reg_0_0);
vst1q_f32(&C[4], C_reg_0_1);
vst1q_f32(&C[8], C_reg_0_2);
vst1q_f32(&C[12], C_reg_1_0);
vst1q_f32(&C[12 + 4], C_reg_1_1);
vst1q_f32(&C[12 + 8], C_reg_1_2);
vst1q_f32(&C[(2) * (12)], C_reg_2_0);
vst1q_f32(&C[(2) * (12) + 4], C_reg_2_1);
vst1q_f32(&C[(2) * (12) + 8], C_reg_2_2);
vst1q_f32(&C[(3) * (12)], C_reg_3_0);
vst1q_f32(&C[(3) * (12) + 4], C_reg_3_1);
vst1q_f32(&C[(3) * (12) + 8], C_reg_3_2);
Ci[0] += C[0];
Ci[1] += C[1];
Ci[2] += C[2];
Ci[3] += C[3];
Ci[4] += C[4];
Ci[5] += C[5];
Ci[6] += C[6];
Ci[7] += C[7];
Ci[8] += C[8];
Ci[ldci] += C[12];
Ci[ldci + 1] += C[13];
Ci[ldci + 2] += C[14];
Ci[ldci + 3] += C[15];
Ci[ldci + 4] += C[16];
Ci[ldci + 5] += C[17];
Ci[ldci + 6] += C[18];
Ci[ldci + 7] += C[19];
Ci[ldci + 8] += C[20];
Ci[2 * ldci] += C[24];
Ci[2 * ldci + 1] += C[25];
Ci[2 * ldci + 2] += C[26];
Ci[2 * ldci + 3] += C[27];
Ci[2 * ldci + 4] += C[28];
Ci[2 * ldci + 5] += C[29];
Ci[2 * ldci + 6] += C[30];
Ci[2 * ldci + 7] += C[31];
Ci[2 * ldci + 8] += C[32];
Ci[3 * ldci] += C[36];
Ci[3 * ldci + 1] += C[37];
Ci[3 * ldci + 2] += C[38];
Ci[3 * ldci + 3] += C[39];
Ci[3 * ldci + 4] += C[40];
Ci[3 * ldci + 5] += C[41];
Ci[3 * ldci + 6] += C[42];
Ci[3 * ldci + 7] += C[43];
Ci[3 * ldci + 8] += C[44];
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
