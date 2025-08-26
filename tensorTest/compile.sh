#!/bin/bash

# compile.sh
# This script compiles the CUDA source file into a shared library.

# --- Configuration ---
# Set this to the architecture of your GPU.
# sm_86: Ampere (e.g., RTX 30 series)
# sm_89: Ada Lovelace (e.g., RTX 40 series)
# sm_90: Hopper (e.g., H100)
GPU_ARCH="sm_86"
OUTPUT_LIB="libtgemm.so"
SOURCE_FILE="tensorcut.cu"

echo "Compiling ${SOURCE_FILE} for arch ${GPU_ARCH} -> ${OUTPUT_LIB}"

nvcc -shared -Xcompiler -fPIC \
     -o ${OUTPUT_LIB} ${SOURCE_FILE} \
     -gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
else
    echo "❌ Compilation failed."
fi