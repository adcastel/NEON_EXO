#include "kernels_NEON_4x4_fp32.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
  A_reg = vld1q_f32(&A[(k) * (4)]);
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
