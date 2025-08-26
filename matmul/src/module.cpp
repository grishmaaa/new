// src/module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// From kernel.cu
extern "C" int matmul_cuda(const float* A, const float* B, float* C, int M, int K, int N);

py::array_t<float> matmul(
    py::array_t<float, py::array::c_style | py::array::forcecast> A,
    py::array_t<float, py::array::c_style | py::array::forcecast> B)
{
    if (A.ndim() != 2 || B.ndim() != 2)
        throw std::runtime_error("A and B must be 2D float32 arrays");

    const int M  = static_cast<int>(A.shape(0));
    const int K  = static_cast<int>(A.shape(1));
    const int K2 = static_cast<int>(B.shape(0));
    const int N  = static_cast<int>(B.shape(1));
    if (K != K2) throw std::runtime_error("Inner dimensions must match (A:MxK, B:KxN)");

    // Allocate output (C-contiguous)
    py::array_t<float> C({M, N});

    // Raw contiguous pointers
    auto a = A.unchecked<2>();
    auto b = B.unchecked<2>();
    auto c = C.mutable_unchecked<2>();

    int ret = matmul_cuda(
        reinterpret_cast<const float*>(a.data(0,0)),
        reinterpret_cast<const float*>(b.data(0,0)),
        reinterpret_cast<float*>(c.mutable_data(0,0)),
        M, K, N
    );
    if (ret != 0)
        throw std::runtime_error("CUDA matmul failed with error code " + std::to_string(ret));
    return C;
}

PYBIND11_MODULE(matmul, m) {
    m.doc() = "Minimal CUDA matmul (importable)";
    m.def("matmul", &matmul, "C = A @ B (float32, GPU)", py::arg("A"), py::arg("B"));
}
