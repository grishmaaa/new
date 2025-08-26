#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "linalg.h"

#define CHECK_OK(expr) do { int _e = (expr); if (_e!=0) { \
  fprintf(stderr, "FAIL: %s => %d\n", #expr, _e); return 1; } } while(0)

static int nearly_eq(float a, float b, float eps){ return fabsf(a-b) <= eps * fmaxf(1.0f, fmaxf(fabsf(a), fabsf(b))); }

int main(void){
  // GEMM sanity: C = A(2x3) * B(3x2)
  float A[2*3] = {1,2,3, 4,5,6}; // ldA=3
  float B[3*2] = {7,8, 9,10, 11,12}; // ldB=2
  float C[2*2] = {0};
  CHECK_OK(linalg_gemm_f32(false,false, 2,2,3, 1.0f, A,3, B,2, 0.0f, C,2, LINALG_ALGO_NAIVE, NULL));
  // Expected [[58,64],[139,154]]
  if (!(C[0]==58 && C[1]==64 && C[2]==139 && C[3]==154)) { fprintf(stderr, "GEMM wrong\n"); return 1; }

  // Bias add (col): add b over N=2
  float bcol[2] = {1, -2};
  CHECK_OK(linalg_bias_add_col_f32(C, bcol, 2,2,2, NULL));
  if (!(C[0]==59 && C[1]==62 && C[2]==140 && C[3]==152)) { fprintf(stderr, "bias col wrong\n"); return 1; }

  // Row/Col sum
  float rowsumN[2]; float colsumM[2];
  CHECK_OK(linalg_row_sum_f32(C,2,2,2, rowsumN, NULL)); // across rows => per col
  CHECK_OK(linalg_col_sum_f32(C,2,2,2, colsumM, NULL));
  if (!(rowsumN[0]==199 && rowsumN[1]==214)) { fprintf(stderr, "row_sum wrong\n"); return 1; }
  if (!(colsumM[0]==121 && colsumM[1]==292)) { fprintf(stderr, "col_sum wrong\n"); return 1; }

  // ReLU fwd/bw
  float z[5] = {-2, -0.1f, 0.0f, 0.1f, 3}; float a[5];
  CHECK_OK(linalg_ewise_relu_f32(z,a,5,NULL));
  if (!(a[0]==0 && a[1]==0 && a[2]==0 && a[3]==0.1f && a[4]==3)) { fprintf(stderr, "relu fwd wrong\n"); return 1; }
  float dA[5] = {1,1,1,1,1}, dZ[5];
  CHECK_OK(linalg_ewise_relu_bw_f32(z,dA,dZ,5,NULL));
  if (!(dZ[0]==0 && dZ[1]==0 && dZ[2]==0 && dZ[3]==1 && dZ[4]==1)) { fprintf(stderr, "relu bw wrong\n"); return 1; }

  // Sigmoid fwd/bw
  float asig[3]; float zs[3] = {-10, 0, 10};
  CHECK_OK(linalg_ewise_sigmoid_f32(zs, asig, 3, NULL));
  if (!(nearly_eq(asig[0], 0.000045f, 1e-2f) && nearly_eq(asig[1], 0.5f, 1e-6f) && nearly_eq(asig[2], 0.999955f, 1e-4f))) {
    fprintf(stderr, "sigmoid fwd wrong: %f %f %f\n", asig[0],asig[1],asig[2]); return 1;
  }
  float dA_s[3] = {1,1,1}, dZ_s[3];
  CHECK_OK(linalg_ewise_sigmoid_bw_f32(asig, dA_s, dZ_s, 3, NULL));

  // Softmax row + CE loss + CE backward
  float logits[2*3] = { 1, 2, 3,   1, -1, 0 }; // M=2, N=3, ld=3
  int32_t labels[2] = {2, 0};
  float P[2*3];
  CHECK_OK(linalg_softmax_row_f32(logits,2,3,3, P,3,NULL));
  float mean_ce;
  CHECK_OK(linalg_ce_loss_from_logits_f32(logits,2,3,3, labels, &mean_ce, NULL));
  if (!nearly_eq(mean_ce, 1.407606f, 1e-4f)) { fprintf(stderr, "CE mean wrong: %f\n", mean_ce); return 1; }
  float dZ_ce[2*3];
  CHECK_OK(linalg_softmax_ce_logits_bw_f32(logits, labels, 2,3,3, dZ_ce, NULL));
  // dZ row 0 should sum to 0
  float s0 = dZ_ce[0]+dZ_ce[1]+dZ_ce[2];
  float s1 = dZ_ce[3]+dZ_ce[4]+dZ_ce[5];
  if (!nearly_eq(s0,0.0f,1e-6f) || !nearly_eq(s1,0.0f,1e-6f)) { fprintf(stderr, "dZ rows not zero-sum\n"); return 1; }

  printf("All tests PASS.\n");
  return 0;
}
