// #include <stdio.h>
// #include <stdlib.h>
// #include <assert.h>

// #include "tiled.h"
// #include "utils.h"

// #define MAX_NUM 10 
// #define MIN_NUM -10 

// int main(int argc, char const *argv[])
// {
//     int N1 = 2678;
//     int N2 = 2678;
//     int N3 = 2678;

//     float* A = (float*)malloc(N1*N2*sizeof(float));
//     for (int i=0; i<N1; i++)
//     {
//         for (int j=0; j<N2; j++)
//             A[i*N2+j] = (float)(rand() % (MAX_NUM - MIN_NUM +1) + MIN_NUM);
//     }

//     float* B = (float*)malloc(N2*N3*sizeof(float));
//     for (int i=0; i<N2; i++)
//     {
//         for (int j=0; j<N3; j++)
//             B[i*N3+j] = (float)(rand() % (MAX_NUM - MIN_NUM +1) + MIN_NUM);
//     }

//     float* C_gpu = (float*)malloc(N1*N3*sizeof(float));
//     double t1_gpu = myCPUTimer();
//     tiled_gpu(A, B, C_gpu, N1, N2, N3);
//     cudaDeviceSynchronize();
//     double t2_gpu = myCPUTimer();

//     double elapsed_ms = (t2_gpu - t1_gpu) / 1000.0;
//     printf("Tiled GPU execution time (N1: %d; N2: %d; N3: %d): %.3f ms\n", N1, N2, N3, elapsed_ms);
//     printf("\n");


//     // Asserting Results
//     // printf("Asserting Results... \n");
//     // for (int i = 0; i < N1; i++)
//     // {
//     //     for (int j = 0; j < N3; j++)
//     //         assert(fabs(C_gpu[i*N3+j] - C_tiled_gpu[i*N3+j]) < 0.00000001);
//     // }
//     // printf("Asserting Passed! \n");

//     // Free memory
//     free(A);
//     free(B);
//     free(C_gpu);

//     return 0;

// }
// ///////////////////////////////////////////////////////////////////////////////////




// main.cu — tiny library shim that reuses your existing implementation
#include "tiled.h"

// Export a C-ABI entry so Python (ctypes) can call it.
// A: (N1 x N2), B: (N2 x N3), C: (N1 x N3), row-major float32
extern "C" int matmul(const float* A, const float* B, float* C,
                      int N1, int N2, int N3)
{
    // Your tiled host wrapper expects non-const float*; we don't modify A/B.
    tiled_gpu(const_cast<float*>(A),
              const_cast<float*>(B),
              C,
              N1, N2, N3);
    return 0; // if you add error handling inside tiled_gpu, propagate it here
}