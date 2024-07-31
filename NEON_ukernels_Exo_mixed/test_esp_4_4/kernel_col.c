#include "kernel_col.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


// gemm_NEON_1x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
free(C);
}

// gemm_NEON_1x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
free(C);
}

// gemm_NEON_1x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
free(C);
}

// gemm_NEON_1x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
free(C);
}

// gemm_NEON_1x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
free(C);
}

// gemm_NEON_1x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
free(C);
}

// gemm_NEON_1x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 1] @DRAM
// )
void gemm_NEON_1x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
free(C);
}

// gemm_NEON_1x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 1] @DRAM
// )
void gemm_NEON_1x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
free(C);
}

// gemm_NEON_2x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
free(C);
}

// gemm_NEON_2x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
free(C);
}

// gemm_NEON_2x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
free(C);
}

// gemm_NEON_2x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
free(C);
}

// gemm_NEON_2x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
free(C);
}

// gemm_NEON_2x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
free(C);
}

// gemm_NEON_2x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 2] @DRAM
// )
void gemm_NEON_2x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
free(C);
}

// gemm_NEON_2x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 2] @DRAM
// )
void gemm_NEON_2x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
free(C);
}

// gemm_NEON_3x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[2] = C[2];
free(C);
}

// gemm_NEON_3x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[2] += C[2];
free(C);
}

// gemm_NEON_3x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[2] = C[2];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[Ci.strides[0] + 2] = C[6];
free(C);
}

// gemm_NEON_3x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[2] += C[2];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[Ci.strides[0] + 2] += C[6];
free(C);
}

// gemm_NEON_3x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[2] = C[2];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[Ci.strides[0] + 2] = C[6];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[2 * Ci.strides[0] + 2] = C[10];
free(C);
}

// gemm_NEON_3x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[2] += C[2];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[Ci.strides[0] + 2] += C[6];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[2 * Ci.strides[0] + 2] += C[10];
free(C);
}

// gemm_NEON_3x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 3] @DRAM
// )
void gemm_NEON_3x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[2] = C[2];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[Ci.strides[0] + 2] = C[6];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[2 * Ci.strides[0] + 2] = C[10];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[3 * Ci.strides[0] + 2] = C[14];
free(C);
}

// gemm_NEON_3x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][4, 3] @DRAM
// )
void gemm_NEON_3x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[2] += C[2];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[Ci.strides[0] + 2] += C[6];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[2 * Ci.strides[0] + 2] += C[10];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[3 * Ci.strides[0] + 2] += C[14];
free(C);
}

// gemm_NEON_4x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[2] = C[2];
Ci.data[3] = C[3];
free(C);
}

// gemm_NEON_4x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[2] += C[2];
Ci.data[3] += C[3];
free(C);
}

// gemm_NEON_4x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[2] = C[2];
Ci.data[3] = C[3];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[Ci.strides[0] + 2] = C[6];
Ci.data[Ci.strides[0] + 3] = C[7];
free(C);
}

// gemm_NEON_4x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[2] += C[2];
Ci.data[3] += C[3];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[Ci.strides[0] + 2] += C[6];
Ci.data[Ci.strides[0] + 3] += C[7];
free(C);
}

// gemm_NEON_4x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[2] = C[2];
Ci.data[3] = C[3];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[Ci.strides[0] + 2] = C[6];
Ci.data[Ci.strides[0] + 3] = C[7];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[2 * Ci.strides[0] + 2] = C[10];
Ci.data[2 * Ci.strides[0] + 3] = C[11];
free(C);
}

// gemm_NEON_4x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(4 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[2] += C[2];
Ci.data[3] += C[3];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[Ci.strides[0] + 2] += C[6];
Ci.data[Ci.strides[0] + 3] += C[7];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[2 * Ci.strides[0] + 2] += C[10];
Ci.data[2 * Ci.strides[0] + 3] += C[11];
free(C);
}

// gemm_NEON_4x4_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C.data[0], C_reg_0_0);
vst1q_s32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_s32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_s32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
}

// gemm_NEON_4x4_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 4] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][4, 4] @DRAM
// )
void gemm_NEON_4x4_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 4
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
C_reg_0_0 = vld1q_s32(&C.data[0]);
C_reg_1_0 = vld1q_s32(&C.data[C.strides[0]]);
C_reg_2_0 = vld1q_s32(&C.data[(2) * (C.strides[0])]);
C_reg_3_0 = vld1q_s32(&C.data[(3) * (C.strides[0])]);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[1];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 4]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
vst1q_s32(&C.data[0], C_reg_0_0);
vst1q_s32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_s32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_s32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
}


/* relying on the following instruction..."
neon_get_low_8xi16(dst,src)
{dst_data} = vget_low_s16(vmovl_s8({src_data}));
*/

/* relying on the following instruction..."
neon_vld_4xi32(dst,src)
{dst_data} = vld1q_s32(&{src_data});
*/

/* relying on the following instruction..."
neon_vld_8xi8(dst,src,e)
{dst_data} = vld1_s8(&{src_data});
*/

/* relying on the following instruction..."
neon_vmlal_8xi16_8xi16(dst,lhs,rhs,jtt)
{dst_data} = vmlal_lane_s16({dst_data}, {lhs_data}, {rhs_data}, {jtt});
*/

/* relying on the following instruction..."
neon_vst_4xi32(dst,src)
vst1q_s32(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
neon_zero_4xi32(dst)
{dst_data} = vmovq_n_s32(0);
*/
