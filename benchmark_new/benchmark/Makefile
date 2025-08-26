CC = nvcc
ARCH ?= sm_86
NVCCFLAGS := -O3 -std=c++14 -arch=$(ARCH)

all: mat_mul

mat_mul: main.cu matmul_cpu.c matmul_gpu.cu tiled.cu tensorcore.cu utils.c
	$(CC) $(NVCCFLAGS) main.cu matmul_cpu.c matmul_gpu.cu tiled.cu tensorcore.cu utils.c -o mat_mul_bench.out

clean: 
	@echo "Removing object files..."
	- rm -f mat_mul_bench.out
