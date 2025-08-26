// // linear_cpu.cpp
// #include <cstddef>
// #include <cmath>
// #include <cstring>
// #include <immintrin.h> // optional; we won't rely on intrinsics here
// #ifdef _OPENMP
// #include <omp.h>
// #endif

// extern "C" int linear_forward_cpu(
//     const float* X,  // M x K
//     const float* W,  // K x N
//     const float* b,  // N (nullable)
//     float* Y,        // M x N (output)
//     int M, int K, int N
// ) {
//     if (!X || !W || !Y || M <= 0 || K <= 0 || N <= 0) return 1;

//     // Y = X * W
//     // Row-major: X[m*K + k], W[k*N + n], Y[m*N + n]
//     #pragma omp parallel for if (M*N > 32768)
//     for (int m = 0; m < M; ++m) {
//         for (int n = 0; n < N; ++n) {
//             float acc = 0.f;
//             const float* x_row = X + m * K;
//             const float* w_col = W + n; // step by N over k
//             for (int k = 0; k < K; ++k) {
//                 acc += x_row[k] * w_col[k * N];
//             }
//             Y[m * N + n] = acc + (b ? b[n] : 0.f);
//         }
//     }
//     return 0;
// }
// linear_cpu.cpp
#include <cstddef>

extern "C" int linear_forward_cpu(
    const float* X,  // M x K
    const float* W,  // K x N
    const float* b,  // N (nullable)
    float* Y,        // M x N
    int M, int K, int N
) {
    if (!X || !W || !Y || M <= 0 || K <= 0 || N <= 0) return 1;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += X[m*K + k] * W[k*N + n];
            Y[m*N + n] = acc + (b ? b[n] : 0.f);
        }
    }
    return 0;
}
