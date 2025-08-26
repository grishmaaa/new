CC = nvcc

all: mat_mul

mat_mul: main.cu tiled.cu utils.c
	@echo "Compiling Square Matrix Multiplication..."
	$(CC) main.cu tiled.cu utils.c -o mat_mul.out

clean: 
	@echo "Removing object files..."
	rm mat_mul.out