# Makefile to compile the C and CUDA files into a shared library

CC = gcc
NVCC = nvcc
NVCC_FLAGS = -shared -Xcompiler -fPIC
CUDA_LIBS = -lcudart -lcublas

TARGET = libtensor.so
C_FILES = tensor_lib.c
CU_FILES = tensor_kernels.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(C_FILES) $(CU_FILES) tensor_lib.h
	$(NVCC) $(NVCC_FLAGS) $(CU_FILES) $(C_FILES) -o $(TARGET) $(CUDA_LIBS)

clean:
	rm -f $(TARGET)
