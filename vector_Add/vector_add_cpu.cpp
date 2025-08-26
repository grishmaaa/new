#include <cstddef>

extern "C" int vector_add_cpu(  // C linkage: prevents C++ name mangling
    const float* A,
    const float* B,
    float* C,
    int N
) {
    if (!A || !B || !C || N <= 0) return 1;
    for (int i = 0; i < N; ++i) C[i] = A[i] + B[i];
    return 0;
}
