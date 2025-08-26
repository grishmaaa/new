#ifndef MAT_MUL_GPU
# define MAT_MUL_GPU

void matmul_kernel(float* A, float* B, float* C, int N1, int N2, int N3);

void matmul_gpu(float* A, float* B, float* C, int N1, int N2, int N3);

#endif