#include "kernel_col.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


// gemm_NEON_1x10_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[9 * Ci.strides[0]] = C[36];
free(C);
}

// gemm_NEON_1x10_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 1] @DRAM
// )
void gemm_NEON_1x10_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[9 * Ci.strides[0]] += C[36];
free(C);
}

// gemm_NEON_1x11_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[10 * Ci.strides[0]] = C[40];
free(C);
}

// gemm_NEON_1x11_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 1] @DRAM
// )
void gemm_NEON_1x11_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[10 * Ci.strides[0]] += C[40];
free(C);
}

// gemm_NEON_1x12_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[10 * Ci.strides[0]] = C[40];
Ci.data[11 * Ci.strides[0]] = C[44];
free(C);
}

// gemm_NEON_1x12_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 1] @DRAM
// )
void gemm_NEON_1x12_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[10 * Ci.strides[0]] += C[40];
Ci.data[11 * Ci.strides[0]] += C[44];
free(C);
}

// gemm_NEON_1x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
free(C);
}

// gemm_NEON_1x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 1] @DRAM
// )
void gemm_NEON_1x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
free(C);
}

// gemm_NEON_1x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
free(C);
}

// gemm_NEON_1x2_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 1] @DRAM
// )
void gemm_NEON_1x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
free(C);
}

// gemm_NEON_1x3_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
free(C);
}

// gemm_NEON_1x3_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 1] @DRAM
// )
void gemm_NEON_1x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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

// gemm_NEON_1x5_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[4 * Ci.strides[0]] = C[16];
free(C);
}

// gemm_NEON_1x5_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 1] @DRAM
// )
void gemm_NEON_1x5_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[4 * Ci.strides[0]] += C[16];
free(C);
}

// gemm_NEON_1x6_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[5 * Ci.strides[0]] = C[20];
free(C);
}

// gemm_NEON_1x6_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 1] @DRAM
// )
void gemm_NEON_1x6_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[5 * Ci.strides[0]] += C[20];
free(C);
}

// gemm_NEON_1x7_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[6 * Ci.strides[0]] = C[24];
free(C);
}

// gemm_NEON_1x7_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 1] @DRAM
// )
void gemm_NEON_1x7_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[6 * Ci.strides[0]] += C[24];
free(C);
}

// gemm_NEON_1x8_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 8
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(8 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[2];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[7 * Ci.strides[0]] = C[28];
free(C);
}

// gemm_NEON_1x8_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 1] @DRAM
// )
void gemm_NEON_1x8_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 8
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(8 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[2];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[7 * Ci.strides[0]] += C[28];
free(C);
}

// gemm_NEON_1x9_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[8 * Ci.strides[0]] = C[32];
free(C);
}

// gemm_NEON_1x9_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 1] @DRAM
// )
void gemm_NEON_1x9_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[8 * Ci.strides[0]] += C[32];
free(C);
}

// gemm_NEON_2x10_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[9 * Ci.strides[0] + 1] = C[37];
free(C);
}

// gemm_NEON_2x10_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 2] @DRAM
// )
void gemm_NEON_2x10_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[9 * Ci.strides[0] + 1] += C[37];
free(C);
}

// gemm_NEON_2x11_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[9 * Ci.strides[0] + 1] = C[37];
Ci.data[10 * Ci.strides[0]] = C[40];
Ci.data[10 * Ci.strides[0] + 1] = C[41];
free(C);
}

// gemm_NEON_2x11_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 2] @DRAM
// )
void gemm_NEON_2x11_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[9 * Ci.strides[0] + 1] += C[37];
Ci.data[10 * Ci.strides[0]] += C[40];
Ci.data[10 * Ci.strides[0] + 1] += C[41];
free(C);
}

// gemm_NEON_2x12_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[9 * Ci.strides[0] + 1] = C[37];
Ci.data[10 * Ci.strides[0]] = C[40];
Ci.data[10 * Ci.strides[0] + 1] = C[41];
Ci.data[11 * Ci.strides[0]] = C[44];
Ci.data[11 * Ci.strides[0] + 1] = C[45];
free(C);
}

// gemm_NEON_2x12_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 2] @DRAM
// )
void gemm_NEON_2x12_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[9 * Ci.strides[0] + 1] += C[37];
Ci.data[10 * Ci.strides[0]] += C[40];
Ci.data[10 * Ci.strides[0] + 1] += C[41];
Ci.data[11 * Ci.strides[0]] += C[44];
Ci.data[11 * Ci.strides[0] + 1] += C[45];
free(C);
}

// gemm_NEON_2x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
free(C);
}

// gemm_NEON_2x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 2] @DRAM
// )
void gemm_NEON_2x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
free(C);
}

// gemm_NEON_2x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 2] @DRAM
// )
void gemm_NEON_2x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 2] @DRAM
// )
void gemm_NEON_2x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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

// gemm_NEON_2x5_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
free(C);
}

// gemm_NEON_2x5_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 2] @DRAM
// )
void gemm_NEON_2x5_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
free(C);
}

// gemm_NEON_2x6_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
free(C);
}

// gemm_NEON_2x6_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 2] @DRAM
// )
void gemm_NEON_2x6_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
free(C);
}

// gemm_NEON_2x7_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
free(C);
}

// gemm_NEON_2x7_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 2] @DRAM
// )
void gemm_NEON_2x7_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
free(C);
}

// gemm_NEON_2x8_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 8
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(8 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[2];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
free(C);
}

// gemm_NEON_2x8_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 2] @DRAM
// )
void gemm_NEON_2x8_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 8
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(8 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[2];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
free(C);
}

// gemm_NEON_2x9_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[Ci.strides[0]] = C[4];
Ci.data[Ci.strides[0] + 1] = C[5];
Ci.data[2 * Ci.strides[0]] = C[8];
Ci.data[2 * Ci.strides[0] + 1] = C[9];
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
free(C);
}

// gemm_NEON_2x9_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 2] @DRAM
// )
void gemm_NEON_2x9_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[Ci.strides[0]] += C[4];
Ci.data[Ci.strides[0] + 1] += C[5];
Ci.data[2 * Ci.strides[0]] += C[8];
Ci.data[2 * Ci.strides[0] + 1] += C[9];
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
free(C);
}

// gemm_NEON_3x10_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[7 * Ci.strides[0] + 2] = C[30];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[8 * Ci.strides[0] + 2] = C[34];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[9 * Ci.strides[0] + 1] = C[37];
Ci.data[9 * Ci.strides[0] + 2] = C[38];
free(C);
}

// gemm_NEON_3x10_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 3] @DRAM
// )
void gemm_NEON_3x10_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[7 * Ci.strides[0] + 2] += C[30];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[8 * Ci.strides[0] + 2] += C[34];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[9 * Ci.strides[0] + 1] += C[37];
Ci.data[9 * Ci.strides[0] + 2] += C[38];
free(C);
}

// gemm_NEON_3x11_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[7 * Ci.strides[0] + 2] = C[30];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[8 * Ci.strides[0] + 2] = C[34];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[9 * Ci.strides[0] + 1] = C[37];
Ci.data[9 * Ci.strides[0] + 2] = C[38];
Ci.data[10 * Ci.strides[0]] = C[40];
Ci.data[10 * Ci.strides[0] + 1] = C[41];
Ci.data[10 * Ci.strides[0] + 2] = C[42];
free(C);
}

// gemm_NEON_3x11_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 3] @DRAM
// )
void gemm_NEON_3x11_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[7 * Ci.strides[0] + 2] += C[30];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[8 * Ci.strides[0] + 2] += C[34];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[9 * Ci.strides[0] + 1] += C[37];
Ci.data[9 * Ci.strides[0] + 2] += C[38];
Ci.data[10 * Ci.strides[0]] += C[40];
Ci.data[10 * Ci.strides[0] + 1] += C[41];
Ci.data[10 * Ci.strides[0] + 2] += C[42];
free(C);
}

// gemm_NEON_3x12_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[7 * Ci.strides[0] + 2] = C[30];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[8 * Ci.strides[0] + 2] = C[34];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[9 * Ci.strides[0] + 1] = C[37];
Ci.data[9 * Ci.strides[0] + 2] = C[38];
Ci.data[10 * Ci.strides[0]] = C[40];
Ci.data[10 * Ci.strides[0] + 1] = C[41];
Ci.data[10 * Ci.strides[0] + 2] = C[42];
Ci.data[11 * Ci.strides[0]] = C[44];
Ci.data[11 * Ci.strides[0] + 1] = C[45];
Ci.data[11 * Ci.strides[0] + 2] = C[46];
free(C);
}

// gemm_NEON_3x12_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][12, 3] @DRAM
// )
void gemm_NEON_3x12_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[7 * Ci.strides[0] + 2] += C[30];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[8 * Ci.strides[0] + 2] += C[34];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[9 * Ci.strides[0] + 1] += C[37];
Ci.data[9 * Ci.strides[0] + 2] += C[38];
Ci.data[10 * Ci.strides[0]] += C[40];
Ci.data[10 * Ci.strides[0] + 1] += C[41];
Ci.data[10 * Ci.strides[0] + 2] += C[42];
Ci.data[11 * Ci.strides[0]] += C[44];
Ci.data[11 * Ci.strides[0] + 1] += C[45];
Ci.data[11 * Ci.strides[0] + 2] += C[46];
free(C);
}

// gemm_NEON_3x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] = C[0];
Ci.data[1] = C[1];
Ci.data[2] = C[2];
free(C);
}

// gemm_NEON_3x1_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 3] @DRAM
// )
void gemm_NEON_3x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
Ci.data[0] += C[0];
Ci.data[1] += C[1];
Ci.data[2] += C[2];
free(C);
}

// gemm_NEON_3x2_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 3] @DRAM
// )
void gemm_NEON_3x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 3] @DRAM
// )
void gemm_NEON_3x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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

// gemm_NEON_3x5_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
free(C);
}

// gemm_NEON_3x5_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 3] @DRAM
// )
void gemm_NEON_3x5_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
free(C);
}

// gemm_NEON_3x6_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
free(C);
}

// gemm_NEON_3x6_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 3] @DRAM
// )
void gemm_NEON_3x6_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
free(C);
}

// gemm_NEON_3x7_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
free(C);
}

// gemm_NEON_3x7_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 3] @DRAM
// )
void gemm_NEON_3x7_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
free(C);
}

// gemm_NEON_3x8_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 8
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(8 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[2];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
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
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[7 * Ci.strides[0] + 2] = C[30];
free(C);
}

// gemm_NEON_3x8_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][8, 3] @DRAM
// )
void gemm_NEON_3x8_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 8
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(8 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[2];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
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
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[7 * Ci.strides[0] + 2] += C[30];
free(C);
}

// gemm_NEON_3x9_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[7 * Ci.strides[0] + 2] = C[30];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[8 * Ci.strides[0] + 2] = C[34];
free(C);
}

// gemm_NEON_3x9_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 3] @DRAM
// )
void gemm_NEON_3x9_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[7 * Ci.strides[0] + 2] += C[30];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[8 * Ci.strides[0] + 2] += C[34];
free(C);
}

// gemm_NEON_4x10_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[3 * Ci.strides[0] + 2] = C[14];
Ci.data[3 * Ci.strides[0] + 3] = C[15];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[4 * Ci.strides[0] + 3] = C[19];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[5 * Ci.strides[0] + 3] = C[23];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[6 * Ci.strides[0] + 3] = C[27];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[7 * Ci.strides[0] + 2] = C[30];
Ci.data[7 * Ci.strides[0] + 3] = C[31];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[8 * Ci.strides[0] + 2] = C[34];
Ci.data[8 * Ci.strides[0] + 3] = C[35];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[9 * Ci.strides[0] + 1] = C[37];
Ci.data[9 * Ci.strides[0] + 2] = C[38];
Ci.data[9 * Ci.strides[0] + 3] = C[39];
free(C);
}

// gemm_NEON_4x10_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][10, 4] @DRAM
// )
void gemm_NEON_4x10_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[3 * Ci.strides[0] + 2] += C[14];
Ci.data[3 * Ci.strides[0] + 3] += C[15];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[4 * Ci.strides[0] + 3] += C[19];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[5 * Ci.strides[0] + 3] += C[23];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[6 * Ci.strides[0] + 3] += C[27];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[7 * Ci.strides[0] + 2] += C[30];
Ci.data[7 * Ci.strides[0] + 3] += C[31];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[8 * Ci.strides[0] + 2] += C[34];
Ci.data[8 * Ci.strides[0] + 3] += C[35];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[9 * Ci.strides[0] + 1] += C[37];
Ci.data[9 * Ci.strides[0] + 2] += C[38];
Ci.data[9 * Ci.strides[0] + 3] += C[39];
free(C);
}

// gemm_NEON_4x11_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[3 * Ci.strides[0] + 2] = C[14];
Ci.data[3 * Ci.strides[0] + 3] = C[15];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[4 * Ci.strides[0] + 3] = C[19];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[5 * Ci.strides[0] + 3] = C[23];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[6 * Ci.strides[0] + 3] = C[27];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[7 * Ci.strides[0] + 2] = C[30];
Ci.data[7 * Ci.strides[0] + 3] = C[31];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[8 * Ci.strides[0] + 2] = C[34];
Ci.data[8 * Ci.strides[0] + 3] = C[35];
Ci.data[9 * Ci.strides[0]] = C[36];
Ci.data[9 * Ci.strides[0] + 1] = C[37];
Ci.data[9 * Ci.strides[0] + 2] = C[38];
Ci.data[9 * Ci.strides[0] + 3] = C[39];
Ci.data[10 * Ci.strides[0]] = C[40];
Ci.data[10 * Ci.strides[0] + 1] = C[41];
Ci.data[10 * Ci.strides[0] + 2] = C[42];
Ci.data[10 * Ci.strides[0] + 3] = C[43];
free(C);
}

// gemm_NEON_4x11_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][11, 4] @DRAM
// )
void gemm_NEON_4x11_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[3 * Ci.strides[0] + 2] += C[14];
Ci.data[3 * Ci.strides[0] + 3] += C[15];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[4 * Ci.strides[0] + 3] += C[19];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[5 * Ci.strides[0] + 3] += C[23];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[6 * Ci.strides[0] + 3] += C[27];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[7 * Ci.strides[0] + 2] += C[30];
Ci.data[7 * Ci.strides[0] + 3] += C[31];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[8 * Ci.strides[0] + 2] += C[34];
Ci.data[8 * Ci.strides[0] + 3] += C[35];
Ci.data[9 * Ci.strides[0]] += C[36];
Ci.data[9 * Ci.strides[0] + 1] += C[37];
Ci.data[9 * Ci.strides[0] + 2] += C[38];
Ci.data[9 * Ci.strides[0] + 3] += C[39];
Ci.data[10 * Ci.strides[0]] += C[40];
Ci.data[10 * Ci.strides[0] + 1] += C[41];
Ci.data[10 * Ci.strides[0] + 2] += C[42];
Ci.data[10 * Ci.strides[0] + 3] += C[43];
free(C);
}

// gemm_NEON_4x12_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C.data[0], C_reg_0_0);
vst1q_s32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_s32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_s32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_s32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_s32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_s32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_s32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_s32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_s32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_s32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_s32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
}

// gemm_NEON_4x12_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][12, 4] @DRAM
// )
void gemm_NEON_4x12_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vld1q_s32(&C.data[0]);
C_reg_1_0 = vld1q_s32(&C.data[C.strides[0]]);
C_reg_2_0 = vld1q_s32(&C.data[(2) * (C.strides[0])]);
C_reg_3_0 = vld1q_s32(&C.data[(3) * (C.strides[0])]);
C_reg_4_0 = vld1q_s32(&C.data[(4) * (C.strides[0])]);
C_reg_5_0 = vld1q_s32(&C.data[(5) * (C.strides[0])]);
C_reg_6_0 = vld1q_s32(&C.data[(6) * (C.strides[0])]);
C_reg_7_0 = vld1q_s32(&C.data[(7) * (C.strides[0])]);
C_reg_8_0 = vld1q_s32(&C.data[(8) * (C.strides[0])]);
C_reg_9_0 = vld1q_s32(&C.data[(9) * (C.strides[0])]);
C_reg_10_0 = vld1q_s32(&C.data[(10) * (C.strides[0])]);
C_reg_11_0 = vld1q_s32(&C.data[(11) * (C.strides[0])]);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C.data[0], C_reg_0_0);
vst1q_s32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_s32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_s32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_s32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_s32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_s32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_s32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
vst1q_s32(&C.data[(8) * (C.strides[0])], C_reg_8_0);
vst1q_s32(&C.data[(9) * (C.strides[0])], C_reg_9_0);
vst1q_s32(&C.data[(10) * (C.strides[0])], C_reg_10_0);
vst1q_s32(&C.data[(11) * (C.strides[0])], C_reg_11_0);
}

// gemm_NEON_4x1_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][1, 4] @DRAM
// )
void gemm_NEON_4x1_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][2, 4] @DRAM
// )
void gemm_NEON_4x2_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][3, 4] @DRAM
// )
void gemm_NEON_4x3_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
// assert stride(B, 0) == 12
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
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
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
// assert stride(B, 0) == 12
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
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
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

// gemm_NEON_4x5_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[3 * Ci.strides[0] + 2] = C[14];
Ci.data[3 * Ci.strides[0] + 3] = C[15];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[4 * Ci.strides[0] + 3] = C[19];
free(C);
}

// gemm_NEON_4x5_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][5, 4] @DRAM
// )
void gemm_NEON_4x5_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[3 * Ci.strides[0] + 2] += C[14];
Ci.data[3 * Ci.strides[0] + 3] += C[15];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[4 * Ci.strides[0] + 3] += C[19];
free(C);
}

// gemm_NEON_4x6_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[3 * Ci.strides[0] + 2] = C[14];
Ci.data[3 * Ci.strides[0] + 3] = C[15];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[4 * Ci.strides[0] + 3] = C[19];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[5 * Ci.strides[0] + 3] = C[23];
free(C);
}

// gemm_NEON_4x6_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][6, 4] @DRAM
// )
void gemm_NEON_4x6_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[3 * Ci.strides[0] + 2] += C[14];
Ci.data[3 * Ci.strides[0] + 3] += C[15];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[4 * Ci.strides[0] + 3] += C[19];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[5 * Ci.strides[0] + 3] += C[23];
free(C);
}

// gemm_NEON_4x7_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[3 * Ci.strides[0] + 2] = C[14];
Ci.data[3 * Ci.strides[0] + 3] = C[15];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[4 * Ci.strides[0] + 3] = C[19];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[5 * Ci.strides[0] + 3] = C[23];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[6 * Ci.strides[0] + 3] = C[27];
free(C);
}

// gemm_NEON_4x7_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][7, 4] @DRAM
// )
void gemm_NEON_4x7_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[3 * Ci.strides[0] + 2] += C[14];
Ci.data[3 * Ci.strides[0] + 3] += C[15];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[4 * Ci.strides[0] + 3] += C[19];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[5 * Ci.strides[0] + 3] += C[23];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[6 * Ci.strides[0] + 3] += C[27];
free(C);
}

// gemm_NEON_4x8_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[2];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
vst1q_s32(&C.data[0], C_reg_0_0);
vst1q_s32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_s32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_s32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_s32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_s32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_s32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_s32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
}

// gemm_NEON_4x8_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 8] @DRAM,
//     beta : i32[1] @DRAM,
//     C : [i32][8, 4] @DRAM
// )
void gemm_NEON_4x8_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* beta, struct exo_win_2i32 C ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(C, 1) == 1
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
C_reg_0_0 = vld1q_s32(&C.data[0]);
C_reg_1_0 = vld1q_s32(&C.data[C.strides[0]]);
C_reg_2_0 = vld1q_s32(&C.data[(2) * (C.strides[0])]);
C_reg_3_0 = vld1q_s32(&C.data[(3) * (C.strides[0])]);
C_reg_4_0 = vld1q_s32(&C.data[(4) * (C.strides[0])]);
C_reg_5_0 = vld1q_s32(&C.data[(5) * (C.strides[0])]);
C_reg_6_0 = vld1q_s32(&C.data[(6) * (C.strides[0])]);
C_reg_7_0 = vld1q_s32(&C.data[(7) * (C.strides[0])]);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[2];
int8x8_t B_temp_0;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
}
vst1q_s32(&C.data[0], C_reg_0_0);
vst1q_s32(&C.data[C.strides[0]], C_reg_1_0);
vst1q_s32(&C.data[(2) * (C.strides[0])], C_reg_2_0);
vst1q_s32(&C.data[(3) * (C.strides[0])], C_reg_3_0);
vst1q_s32(&C.data[(4) * (C.strides[0])], C_reg_4_0);
vst1q_s32(&C.data[(5) * (C.strides[0])], C_reg_5_0);
vst1q_s32(&C.data[(6) * (C.strides[0])], C_reg_6_0);
vst1q_s32(&C.data[(7) * (C.strides[0])], C_reg_7_0);
}

// gemm_NEON_4x9_b0_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b0_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] = C[12];
Ci.data[3 * Ci.strides[0] + 1] = C[13];
Ci.data[3 * Ci.strides[0] + 2] = C[14];
Ci.data[3 * Ci.strides[0] + 3] = C[15];
Ci.data[4 * Ci.strides[0]] = C[16];
Ci.data[4 * Ci.strides[0] + 1] = C[17];
Ci.data[4 * Ci.strides[0] + 2] = C[18];
Ci.data[4 * Ci.strides[0] + 3] = C[19];
Ci.data[5 * Ci.strides[0]] = C[20];
Ci.data[5 * Ci.strides[0] + 1] = C[21];
Ci.data[5 * Ci.strides[0] + 2] = C[22];
Ci.data[5 * Ci.strides[0] + 3] = C[23];
Ci.data[6 * Ci.strides[0]] = C[24];
Ci.data[6 * Ci.strides[0] + 1] = C[25];
Ci.data[6 * Ci.strides[0] + 2] = C[26];
Ci.data[6 * Ci.strides[0] + 3] = C[27];
Ci.data[7 * Ci.strides[0]] = C[28];
Ci.data[7 * Ci.strides[0] + 1] = C[29];
Ci.data[7 * Ci.strides[0] + 2] = C[30];
Ci.data[7 * Ci.strides[0] + 3] = C[31];
Ci.data[8 * Ci.strides[0]] = C[32];
Ci.data[8 * Ci.strides[0] + 1] = C[33];
Ci.data[8 * Ci.strides[0] + 2] = C[34];
Ci.data[8 * Ci.strides[0] + 3] = C[35];
free(C);
}

// gemm_NEON_4x9_b1_col_i32(
//     KC : size,
//     alpha : i32[1] @DRAM,
//     A : [i8][KC, 4] @DRAM,
//     B : [i8][KC, 12] @DRAM,
//     b : i32[1] @DRAM,
//     Ci : [i32][9, 4] @DRAM
// )
void gemm_NEON_4x9_b1_col_i32( void *ctxt, int_fast32_t KC, const int32_t* alpha, struct exo_win_2i8c A, struct exo_win_2i8c B, const int32_t* b, struct exo_win_2i32 Ci ) {
// assert stride(A, 0) == 4
// assert stride(A, 1) == 1
// assert stride(B, 0) == 12
// assert stride(B, 1) == 1
// assert stride(Ci, 1) == 1
int32_t *C = (int32_t*) malloc(12 * 4 * sizeof(*C));
int32x4_t C_reg_0_0;
int32x4_t C_reg_1_0;
int32x4_t C_reg_2_0;
int32x4_t C_reg_3_0;
int32x4_t C_reg_4_0;
int32x4_t C_reg_5_0;
int32x4_t C_reg_6_0;
int32x4_t C_reg_7_0;
int32x4_t C_reg_8_0;
int32x4_t C_reg_9_0;
int32x4_t C_reg_10_0;
int32x4_t C_reg_11_0;
C_reg_0_0 = vmovq_n_s32(0);
C_reg_1_0 = vmovq_n_s32(0);
C_reg_2_0 = vmovq_n_s32(0);
C_reg_3_0 = vmovq_n_s32(0);
C_reg_4_0 = vmovq_n_s32(0);
C_reg_5_0 = vmovq_n_s32(0);
C_reg_6_0 = vmovq_n_s32(0);
C_reg_7_0 = vmovq_n_s32(0);
C_reg_8_0 = vmovq_n_s32(0);
C_reg_9_0 = vmovq_n_s32(0);
C_reg_10_0 = vmovq_n_s32(0);
C_reg_11_0 = vmovq_n_s32(0);
int16x4_t A_reg_0;
int8x8_t A_temp_0;
int16x4_t B_reg[3];
int8x8_t B_temp_0;
int8x8_t B_temp_1;
for (int_fast32_t kt = 0; kt < ((KC) / (4)); kt++) {
  A_temp_0 = vld1_s8(&A.data[(4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(1 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(1 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(1 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(2 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(2 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(2 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
  A_temp_0 = vld1_s8(&A.data[(3 + 4 * kt) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(3 + 4 * kt) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(3 + 4 * kt) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
for (int_fast32_t ktt = 0; ktt < KC % 4; ktt++) {
  A_temp_0 = vld1_s8(&A.data[(ktt + (KC / 4) * 4) * 4]);
  A_reg_0 = vget_low_s16(vmovl_s8(A_temp_0));
  B_temp_0 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12)]);
  B_temp_1 = vld1_s8(&B.data[(ktt + (KC / 4) * 4) * (12) + 8]);
  B_reg[0] = vget_low_s16(vmovl_s8(B_temp_0));
  B_reg[1] = vget_high_s16(vmovl_s8(B_temp_0));
  B_reg[2] = vget_low_s16(vmovl_s8(B_temp_1));
  C_reg_0_0 = vmlal_lane_s16(C_reg_0_0, A_reg_0, B_reg[0], (0));
  C_reg_1_0 = vmlal_lane_s16(C_reg_1_0, A_reg_0, B_reg[0], (1));
  C_reg_2_0 = vmlal_lane_s16(C_reg_2_0, A_reg_0, B_reg[0], (2));
  C_reg_3_0 = vmlal_lane_s16(C_reg_3_0, A_reg_0, B_reg[0], (3));
  C_reg_4_0 = vmlal_lane_s16(C_reg_4_0, A_reg_0, B_reg[1], (0));
  C_reg_5_0 = vmlal_lane_s16(C_reg_5_0, A_reg_0, B_reg[1], (1));
  C_reg_6_0 = vmlal_lane_s16(C_reg_6_0, A_reg_0, B_reg[1], (2));
  C_reg_7_0 = vmlal_lane_s16(C_reg_7_0, A_reg_0, B_reg[1], (3));
  C_reg_8_0 = vmlal_lane_s16(C_reg_8_0, A_reg_0, B_reg[2], (0));
  C_reg_9_0 = vmlal_lane_s16(C_reg_9_0, A_reg_0, B_reg[2], (1));
  C_reg_10_0 = vmlal_lane_s16(C_reg_10_0, A_reg_0, B_reg[2], (2));
  C_reg_11_0 = vmlal_lane_s16(C_reg_11_0, A_reg_0, B_reg[2], (3));
}
vst1q_s32(&C[0], C_reg_0_0);
vst1q_s32(&C[4], C_reg_1_0);
vst1q_s32(&C[(2) * 4], C_reg_2_0);
vst1q_s32(&C[(3) * 4], C_reg_3_0);
vst1q_s32(&C[(4) * 4], C_reg_4_0);
vst1q_s32(&C[(5) * 4], C_reg_5_0);
vst1q_s32(&C[(6) * 4], C_reg_6_0);
vst1q_s32(&C[(7) * 4], C_reg_7_0);
vst1q_s32(&C[(8) * 4], C_reg_8_0);
vst1q_s32(&C[(9) * 4], C_reg_9_0);
vst1q_s32(&C[(10) * 4], C_reg_10_0);
vst1q_s32(&C[(11) * 4], C_reg_11_0);
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
Ci.data[3 * Ci.strides[0]] += C[12];
Ci.data[3 * Ci.strides[0] + 1] += C[13];
Ci.data[3 * Ci.strides[0] + 2] += C[14];
Ci.data[3 * Ci.strides[0] + 3] += C[15];
Ci.data[4 * Ci.strides[0]] += C[16];
Ci.data[4 * Ci.strides[0] + 1] += C[17];
Ci.data[4 * Ci.strides[0] + 2] += C[18];
Ci.data[4 * Ci.strides[0] + 3] += C[19];
Ci.data[5 * Ci.strides[0]] += C[20];
Ci.data[5 * Ci.strides[0] + 1] += C[21];
Ci.data[5 * Ci.strides[0] + 2] += C[22];
Ci.data[5 * Ci.strides[0] + 3] += C[23];
Ci.data[6 * Ci.strides[0]] += C[24];
Ci.data[6 * Ci.strides[0] + 1] += C[25];
Ci.data[6 * Ci.strides[0] + 2] += C[26];
Ci.data[6 * Ci.strides[0] + 3] += C[27];
Ci.data[7 * Ci.strides[0]] += C[28];
Ci.data[7 * Ci.strides[0] + 1] += C[29];
Ci.data[7 * Ci.strides[0] + 2] += C[30];
Ci.data[7 * Ci.strides[0] + 3] += C[31];
Ci.data[8 * Ci.strides[0]] += C[32];
Ci.data[8 * Ci.strides[0] + 1] += C[33];
Ci.data[8 * Ci.strides[0] + 2] += C[34];
Ci.data[8 * Ci.strides[0] + 3] += C[35];
free(C);
}


/* relying on the following instruction..."
neon_get_high_8xi16(dst,src)
{dst_data} = vget_high_s16(vmovl_s8({src_data}));
*/

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
