CC = nvcc

all: mat_mul

mat_mul: main.cu matmul_cpu.c matmul_gpu.cu tiled.cu utils.c
	
	$(CC) main.cu matmul_cpu.c matmul_gpu.cu tiled.cu utils.c -o mat_mul_bench.out

clean: 
	@echo "Removing object files..."
	rm mat_mul_bench.out