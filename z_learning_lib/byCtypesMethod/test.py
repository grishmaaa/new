import ctypes

testlib = ctypes.CDLL('./lib.so')
testlib.factorial.restype= ctypes.c_int
testlib.calculate_e.restype= ctypes.c_double
testlib.calculate_e.argtypes= [ctypes.c_int]


testlib.hello()

print("Factorial of 5 is:", testlib.factorial(5))
print("Value of e is approximately:", testlib.calculate_e(10))