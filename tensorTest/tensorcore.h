// tensorcore.h
#pragma once

#ifdef __cplusplus
  // C++ / CUDA side: we can include CUDA headers & use half
  #include <cuda_fp16.h>
  extern "C" {
    // Accept device pointers; sizes are row-major: C(MxN)=A(MxK)*B(KxN)
    void naive_tensor_tgemm(half *d_A_ptr,
                            half *d_B_ptr,
                            float *d_C_ptr,
                            int C_n_rows, int C_n_cols, int A_n_cols);
  }
#else
  // C side: don't mention 'half' (not a C type). Use opaque pointers.
  #include <stdint.h>
  void naive_tensor_tgemm(void *d_A_ptr,
                          void *d_B_ptr,
                          float *d_C_ptr,
                          int C_n_rows, int C_n_cols, int A_n_cols);
#endif
