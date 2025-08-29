from distutils.core import setup, Extension

def main():
    setup(
        name="testlib",
        version="1.0.0",
        description="A simple example package",
        author="grishma",
        author_email="grishma.renuka@gmail.com",
        # ext_modules=[Extension("mymodule", sources=["testlib.c"])],
        ext_modules=[Extension("mymodule", sources=["fastmatmul1.cpython-310-x86_64-linux-gnu.so"])],

    )

if __name__ == "__main__":
    main()