from pathlib import Path
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11, numpy as np

def find_cuda_home():
    for k in ("CUDA_HOME", "CUDA_PATH"):
        if k in os.environ:
            return os.environ[k]
    default = "/usr/local/cuda"
    if Path(default).exists():
        return default
    raise RuntimeError("CUDA not found. Set CUDA_HOME or install to /usr/local/cuda")

class BuildExt(build_ext):
    def build_extension(self, ext):
        cuda_home = find_cuda_home()
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        cu_sources = [s for s in ext.sources if s.endswith(".cu")]
        cpp_sources = [s for s in ext.sources if not s.endswith(".cu")]
        ext.sources = cpp_sources  # leave only C/C++ sources for distutils

        # compile CUDA sources with nvcc
        cu_objects = []
        for src in cu_sources:
            srcp = Path(src)
            objp = build_temp / (srcp.stem + ".o")
            cmd = [
                nvcc, "-c", str(srcp),
                "-o", str(objp),
                "-O3", "-Xcompiler", "-fPIC", "-std=c++14",
            ]
            for inc in ext.include_dirs:
                cmd += ["-I", inc]
            self.spawn(cmd)
            cu_objects.append(str(objp))

        # link settings so libcudart is found
        libdir = os.path.join(cuda_home, "lib64")
        ext.library_dirs = list(set((ext.library_dirs or []) + [libdir]))
        ext.libraries = list(set((ext.libraries or []) + ["cudart"]))
        # helps loader find libcudart at runtime on Linux
        ext.runtime_library_dirs = (ext.runtime_library_dirs or []) + [libdir]

        # pass nvcc-built objects to the normal linker
        ext.extra_objects = (ext.extra_objects or []) + cu_objects

        super().build_extension(ext)

ext = Extension(
    name="matmul",
    sources=["src/module.cpp", "src/kernel.cu"],
    include_dirs=[pybind11.get_include(), np.get_include()],
    language="c++",
    # 🔧 was a dict before; must be a LIST for the host C++ compiler
    extra_compile_args=["-O3", "-std=c++14"],
)

setup(
    name="matmul",
    version="0.0.1",
    description="Minimal CUDA matmul (pybind11 importable)",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)
