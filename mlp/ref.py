#!/usr/bin/env python3
import argparse, numpy as np, os
from pathlib import Path
from mlp import MLP

def load_vector(path: str):
    txt = Path(path).read_text().strip().replace(",", " ")
    vals = [float(x) for x in txt.split() if x]
    if len(vals) != 784:
        raise ValueError(f"Expected 784 values, got {len(vals)}")
    x = np.array(vals, dtype=np.float32).reshape(784,1)
    if x.max() > 1.0: x = x/255.0
    return x

def load_image(path: str):
    try:
        from PIL import Image
    except Exception:
        raise RuntimeError("Install Pillow: pip install pillow")
    img = Image.open(path).convert("L").resize((28,28))
    arr = np.array(img).astype(np.float32)/255.0
    return arr.reshape(784,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=str, default="128,64")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--load", type=str, required=True, help="weights .npz saved from your MLP")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="path to digit image")
    src.add_argument("--vector", type=str, help="path to 784-length txt/csv")
    args = ap.parse_args()

    hidden = [int(h) for h in args.hidden.split(",") if h.strip()]
    model = MLP(784, hidden, 10, weight_initialization="he_normal",
                activation_func="relu", loss_function="cross_entropy_loss",
                learning_rate=args.lr)

    data = np.load(args.load)
    L = len(model.layer_sizes)-1
    model.weights = [data[f"W{i+1}"] for i in range(L)]
    model.biases  = [data[f"b{i+1}"] for i in range(L)]

    x = load_image(args.image) if args.image else load_vector(args.vector)
    probs = model.forward(x)             # (10,1)
    pred = int(np.argmax(probs, axis=0)[0])
    top3 = [(int(i), float(probs[i,0])) for i in np.argsort(-probs[:,0])[:3]]

    print("pred:", pred)
    # print("top-3:", ", ".join(f"{c}:{p:.3f}" for c,p in top3))

if __name__ == "__main__":
    main()
