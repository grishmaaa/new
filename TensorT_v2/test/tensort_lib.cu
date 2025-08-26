// tensort_lib.cu — CUDA kernels + C wrappers for TensorT (float32)
#include "tensort_lib.h"
#include <string.h>

// ---------- helpers ----------
__host__ __device__ inline long long _prod_ll(const int* a, int n){
    long long p = 1; for (int i=0;i<n;++i) p *= (long long)a[i]; return p;
}

static void _fill_strides_rowmajor(int* strides, const int* shape, int ndim){
    long long s = 1;
    for (int i=ndim-1; i>=0; --i) {
        strides[i] = (int)s;
        s *= shape[i];
    }
}

// ---- scalar ew kernels ----
__global__ void ew_add_s(const float* a, float s, float* c, long long n){ long long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]+s; }
__global__ void ew_sub_s(const float* a, float s, float* c, long long n){ long long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]-s; }
__global__ void ew_rsub_s(float s, const float* a, float* c, long long n){ long long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=s-a[i]; }
__global__ void ew_mul_s(const float* a, float s, float* c, long long n){ long long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]*s; }
__global__ void ew_div_s(const float* a, float s, float* c, long long n){ long long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=a[i]/s; }
__global__ void ew_rdiv_s(float s, const float* a, float* c, long long n){ long long i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=s/a[i]; }

static TT_Tensor* _alloc_like_gpu(const TT_Tensor* x); // already present

static TT_Tensor* _launch1s(void(*k)(const float*,float,float*,long long),
                            const TT_Tensor* a, float s){
    const int TPB=256; long long n=a->size; int blocks=(int)((n+TPB-1)/TPB);
    TT_Tensor* out=_alloc_like_gpu(a);
    k<<<blocks,TPB>>>(a->data,s,out->data,n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}
static TT_Tensor* _launch1rs(void(*k)(float,const float*,float*,long long),
                             float s, const TT_Tensor* a){
    const int TPB=256; long long n=a->size; int blocks=(int)((n+TPB-1)/TPB);
    TT_Tensor* out=_alloc_like_gpu(a);
    k<<<blocks,TPB>>>(s,a->data,out->data,n);
    CUDA_CHECK(cudaGetLastError());
    return out;
}

extern "C" TT_Tensor* tt_add_scalar_gpu(const TT_Tensor* a, float s){ return _launch1s(ew_add_s,a,s); }
extern "C" TT_Tensor* tt_sub_scalar_gpu(const TT_Tensor* a, float s){ return _launch1s(ew_sub_s,a,s); }
extern "C" TT_Tensor* tt_rsub_scalar_gpu(float s, const TT_Tensor* a){ return _launch1rs(ew_rsub_s,s,a); }
extern "C" TT_Tensor* tt_mul_scalar_gpu(const TT_Tensor* a, float s){ return _launch1s(ew_mul_s,a,s); }
extern "C" TT_Tensor* tt_div_scalar_gpu(const TT_Tensor* a, float s){ return _launch1s(ew_div_s,a,s); }
extern "C" TT_Tensor* tt_rdiv_scalar_gpu(float s, const TT_Tensor* a){ return _launch1rs(ew_rdiv_s,s,a); }


extern "C" long long tt_product(const int* a, int n){ return _prod_ll(a,n); }

// ---------- host/device alloc ----------
extern "C" TT_Tensor* tt_tensor_init_host(const int* shape, int ndim){
    TT_Tensor* t = (TT_Tensor*)malloc(sizeof(TT_Tensor));
    t->ndim = ndim;
    t->shape   = (int*)malloc(sizeof(int)*ndim);
    t->strides = (int*)malloc(sizeof(int)*ndim);
    memcpy(t->shape, shape, sizeof(int)*ndim);
    _fill_strides_rowmajor(t->strides, t->shape, ndim);
    t->size = _prod_ll(t->shape, ndim);
    t->data = (float*)malloc(sizeof(float)*t->size);
    return t;
}

extern "C" TT_Tensor* tt_tensor_init_gpu(const int* shape, int ndim){
    TT_Tensor* t = (TT_Tensor*)malloc(sizeof(TT_Tensor));
    t->ndim = ndim;
    t->shape   = (int*)malloc(sizeof(int)*ndim);
    t->strides = (int*)malloc(sizeof(int)*ndim);
    memcpy(t->shape, shape, sizeof(int)*ndim);
    _fill_strides_rowmajor(t->strides, t->shape, ndim);
    t->size = _prod_ll(t->shape, ndim);
    CUDA_CHECK(cudaMalloc((void**)&t->data, sizeof(float)*t->size));
    return t;
}

extern "C" void tt_tensor_free_host(TT_Tensor* t){
    if(!t) return;
    free(t->shape); free(t->strides); free(t->data); free(t);
}

extern "C" void tt_tensor_free_gpu(TT_Tensor* t){
    if(!t) return;
    CUDA_CHECK(cudaFree(t->data));
    free(t->shape); free(t->strides); free(t);
}

extern "C" void tt_copy_h2d(const TT_Tensor* h_src, TT_Tensor* d_dst){
    // assumes identical shapes
    CUDA_CHECK(cudaMemcpy(d_dst->data, h_src->data, sizeof(float)*h_src->size, cudaMemcpyHostToDevice));
}

extern "C" void tt_copy_d2h(const TT_Tensor* d_src, TT_Tensor* h_dst){
    CUDA_CHECK(cudaMemcpy(h_dst->data, d_src->data, sizeof(float)*d_src->size, cudaMemcpyDeviceToHost));
}

// ---------- elementwise kernels ----------
__global__ void ew_add(const float* a, const float* b, float* c, long long n){
    long long i = blockIdx.x*blockDim.x + threadIdx.x; if(i<n) c[i] = a[i] + b[i]; }
__global__ void ew_sub(const float* a, const float* b, float* c, long long n){
    long long i = blockIdx.x*blockDim.x + threadIdx.x; if(i<n) c[i] = a[i] - b[i]; }
__global__ void ew_mul(const float* a, const float* b, float* c, long long n){
    long long i = blockIdx.x*blockDim.x + threadIdx.x; if(i<n) c[i] = a[i] * b[i]; }
__global__ void ew_div(const float* a, const float* b, float* c, long long n){
    long long i = blockIdx.x*blockDim.x + threadIdx.x; if(i<n) c[i] = a[i] / b[i]; }

static TT_Tensor* _alloc_like_gpu(const TT_Tensor* x){
    return tt_tensor_init_gpu(x->shape, x->ndim);
}

static void _launch1(void(*k)(const float*,const float*,float*,long long), const TT_Tensor* a, const TT_Tensor* b, TT_Tensor* out){
    const int TPB = 256; long long n = a->size; int blocks = (int)((n + TPB - 1)/TPB);
    k<<<blocks, TPB>>>(a->data, b->data, out->data, n);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" TT_Tensor* tt_add_gpu(const TT_Tensor* a, const TT_Tensor* b){
    if (a->size != b->size) return NULL; TT_Tensor* out = _alloc_like_gpu(a); _launch1(ew_add, a,b,out); return out; }
extern "C" TT_Tensor* tt_sub_gpu(const TT_Tensor* a, const TT_Tensor* b){
    if (a->size != b->size) return NULL; TT_Tensor* out = _alloc_like_gpu(a); _launch1(ew_sub, a,b,out); return out; }
extern "C" TT_Tensor* tt_mul_gpu(const TT_Tensor* a, const TT_Tensor* b){
    if (a->size != b->size) return NULL; TT_Tensor* out = _alloc_like_gpu(a); _launch1(ew_mul, a,b,out); return out; }
extern "C" TT_Tensor* tt_div_gpu(const TT_Tensor* a, const TT_Tensor* b){
    if (a->size != b->size) return NULL; TT_Tensor* out = _alloc_like_gpu(a); _launch1(ew_div, a,b,out); return out; }

// ---------- 2D matmul ----------
__global__ void mm2d(const float* A, const float* B, float* C, int M, int K, int N){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row<M && col<N){
        float acc = 0.f;
        for (int k=0;k<K;++k){ acc += A[row*K + k] * B[k*N + col]; }
        C[row*N + col] = acc;
    }
}

extern "C" TT_Tensor* tt_matmul2d_gpu(const TT_Tensor* A, const TT_Tensor* B){
    if (A->ndim != 2 || B->ndim != 2) return NULL;
    int M = A->shape[0], K = A->shape[1];
    int Kb= B->shape[0], N = B->shape[1];
    if (K != Kb) return NULL;
    int shape_out[2] = {M, N};
    TT_Tensor* C = tt_tensor_init_gpu(shape_out, 2);
    dim3 block(16,16); dim3 grid((N+15)/16, (M+15)/16);
    mm2d<<<grid, block>>>(A->data, B->data, C->data, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    return C;
}

// ---------- debug print ----------
static void _print_impl(float* data, const int* shape, int ndim, int depth){
    if (ndim==0){ printf("%.4f", *data); return; }
    printf("[");
    long long stride = 1; for (int i=1;i<ndim;i++) stride *= shape[i];
    for (int i=0;i<shape[0]; ++i){
        if (i>0) printf(", ");
        _print_impl(data + i*stride, shape+1, ndim-1, depth+1);
    }
    printf("]");
}

extern "C" void tt_print_host(const TT_Tensor* t){
    if (!t){ printf("<null>\n"); return; }
    printf("tensor(shape=[");
    for (int i=0;i<t->ndim;i++){ printf("%d%s", t->shape[i], (i==t->ndim-1)?"] ":", "); }
    printf("ndim=%d)\n", t->ndim);
    _print_impl(t->data, t->shape, t->ndim, 0);
    printf("\n");
}