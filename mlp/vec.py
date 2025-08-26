# import numpy as np

# # make a fake "digit 1" image
# img = np.zeros((28,28), dtype=int)
# img[:, 14:16] = 1   # vertical stroke in the middle

# # save to vec.txt
# flat = img.flatten()
# np.savetxt("vec2.txt", flat, fmt="%d")

# make_vecs.py
import numpy as np

def make_one(col=14, thickness=2, noise=0.0):
    img = np.zeros((28, 28), dtype=float)
    img[:, col:col+thickness] = 1.0
    if noise > 0:
        img += noise * np.random.rand(28, 28)
        img = np.clip(img, 0, 1)
    return img.flatten()

def save_vec(path, vec):
    np.savetxt(path, vec, fmt="%.6f")

if __name__ == "__main__":
    # generate 5 variants, columns 12..16
    for i, col in enumerate([12, 13, 14, 15, 16], start=1):
        v = make_one(col=col, thickness=2, noise=0.0)
        save_vec(f"vec_{i}.txt", v)
    print("Wrote:", ", ".join([f"vec_{i}.txt" for i in range(1,6)]))
