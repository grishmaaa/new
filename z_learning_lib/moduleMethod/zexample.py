import mymodule

powers = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536]
for p in powers:
    log2x =mymodule.logarithm_2(p)
    print(f"Logarithm base 2 of {p} is {log2x}")