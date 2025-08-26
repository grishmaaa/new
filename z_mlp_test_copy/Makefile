# # Makefile to compile the C and CUDA files into a shared library

# CC = gcc
# NVCC = nvcc
# NVCC_FLAGS = -shared -Xcompiler -fPIC
# CUDA_LIBS = -lcudart -lcublas

# TARGET = libtensor.so
# C_FILES = tensor_cpu_ops.c
# CU_FILES = tensor_cuda_kernels.cu

# .PHONY: all clean

# all: $(TARGET)

# $(TARGET): $(C_FILES) $(CU_FILES) tensor.h
# 	$(NVCC) $(NVCC_FLAGS) $(CU_FILES) $(C_FILES) -o $(TARGET) $(CUDA_LIBS)

# clean:
# 	rm -f $(TARGET)
# ######################################################################################################
# Makefile for building the CUDA-accelerated Tensor library

# Compiler for CUDA C++ files
NVCC = nvcc
NVCC_FLAGS := -ccbin g++-10
# Compiler for standard C files
GCC = gcc

# Standard compilation flags for shared library
CFLAGS = -Xcompiler -fPIC

# Flags for linking shared library
LDFLAGS = -shared -lcudart -lcublas

# Source files
CUDA_SRC = tensor_cuda_kernels.cu
C_SRC = tensor_cpu_ops.c

# Object files (CUDA files become .o, C files become .o)
CUDA_OBJ = $(CUDA_SRC:.cu=.o)
C_OBJ = $(C_SRC:.c=.o)

# Output library name
TARGET = libtensor.so

all: $(TARGET)

$(TARGET): $(CUDA_OBJ) $(C_OBJ)
	$(NVCC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CFLAGS) -c $< -o $@

$(C_OBJ): $(C_SRC)
	$(GCC) -fPIC -c $< -o $@

clean:
	rm -f $(CUDA_OBJ) $(C_OBJ) $(TARGET)