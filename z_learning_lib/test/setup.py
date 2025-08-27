# setup.py
from distutils.core import setup, Extension

def main():
    setup(
        name="fastmatmul",
        version="1.0.0",
        description="Fast matmul (PyObjects + C ABI)",
        author="grishma",
        author_email="grishma.renuka@gmail.com",
        ext_modules=[Extension("fastmatmul", sources=["fastmatmul1.c"])],
    )

if __name__ == "__main__":
    main()
