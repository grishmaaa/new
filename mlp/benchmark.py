import argparse
import time
import sys
# import os
import numpy as np
from pathlib import Path
# --- Perf knobs (TF & Torch) -----------------------------------------------
import os

def _configure_runtime(threads: int | None, tf_onednn: int | None):
    # Respect explicit env if already set; otherwise set sensible defaults.
    if threads is not None:
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(threads))
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    if tf_onednn is not None:
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = str(tf_onednn)  # "1" fast, "0" plain

    # Apply to Torch if present
    try:
        import torch
        if threads is not None:
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
            torch.set_num_interop_threads(int(os.environ.get("TF_NUM_INTEROP_THREADS", "1")))
    except Exception:
        pass

    # Apply to TensorFlow if present
    try:
        import tensorflow as tf
        # Set TF threading explicitly (works on CPU path)
        if threads is not None:
            tf.config.threading.set_intra_op_parallelism_threads(int(os.environ["TF_NUM_INTRAOP_THREADS"]))
            tf.config.threading.set_inter_op_parallelism_threads(int(os.environ["TF_NUM_INTEROP_THREADS"]))
    except Exception:
        pass
# ---------------------------------------------------------------------------
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")          # tune: set to your CPU cores
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")    # keep TF optimized; set to "0" to disable

try:
    import torch
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(1)
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:
    from mlp import MLP as MLP
except Exception as e:
    print("ERROR: Could not import your MLP from mlp.py. Make sure benchmark.py sits next to mlp.py.")
    print("Import error:", repr(e))
    sys.exit(1)

def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def load_mnist(subset: int = 10000, seed: int = 42):
    set_seed(seed)
    X_train = X_test = y_train = y_test = None

    # Try TensorFlow first
    try:
        import tensorflow as tf
        (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
        X_train = xtr.reshape(-1, 28*28).astype(np.float64) / 255.0
        X_test  = xte.reshape(-1, 28*28).astype(np.float64) / 255.0
        y_train = ytr.astype(np.int64)
        y_test  = yte.astype(np.int64)
    except Exception:
        # Fallback: torchvision
        try:
            from torchvision import datasets, transforms
            ds_train = datasets.MNIST(root=str(HERE / "data"), train=True, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
            ds_test  = datasets.MNIST(root=str(HERE / "data"), train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))
            X_train = ds_train.data.numpy().reshape(-1, 28*28).astype(np.float64) / 255.0
            y_train = ds_train.targets.numpy().astype(np.int64)
            X_test  = ds_test.data.numpy().reshape(-1, 28*28).astype(np.float64) / 255.0
            y_test  = ds_test.targets.numpy().astype(np.int64)
        except Exception as e:
            raise RuntimeError(
                "Could not load MNIST via TensorFlow or torchvision. "
                "Install either 'tensorflow' or 'torch torchvision'."
            ) from e

    rng = np.random.default_rng(seed)
    if subset and subset < len(X_train):
        idx = rng.choice(len(X_train), size=subset, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    if subset and subset < len(X_test):
        idx = rng.choice(len(X_test), size=min(subset//5, len(X_test)), replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    X_train_T = X_train.T  # (784, m)
    X_test_T  = X_test.T

    def one_hot(y, num_classes=10):
        Y = np.zeros((num_classes, y.shape[0]), dtype=np.float64)
        Y[y, np.arange(y.shape[0])] = 1.0
        return Y

    Y_train_oh = one_hot(y_train, 10)
    Y_test_oh  = one_hot(y_test, 10)
    return X_train_T, Y_train_oh, y_train, X_test_T, Y_test_oh, y_test

def accuracy_from_user_pred(user_pred, y_true_int):
    """
    user_pred: either (C, m) probabilities/logits OR (m,) class indices.
    y_true_int: (m,) ints
    """
    if user_pred.ndim == 1:
        preds = user_pred.astype(np.int64)
    elif user_pred.ndim == 2:
        preds = np.argmax(user_pred, axis=0)
    else:
        raise ValueError(f"Unsupported predict() output shape: {user_pred.shape}")
    return float((preds == y_true_int).mean())


def run_user_mlp(X_train, Y_train_oh, X_test, y_test_int, hidden, lr, epochs, seed):
    hidden_layers = [int(h) for h in (hidden.split(",") if isinstance(hidden, str) else hidden) if h.strip()]
    model = MLP(
        input_size=784,
        hidden_layers=hidden_layers,
        output_size=10,
        weight_initialization='he_normal',
        activation_func='relu',
        loss_function='cross_entropy_loss',
        learning_rate=lr
    )
    t0 = time.time()
    model.train(X_train, Y_train_oh, epochs=epochs, print_cost_every=max(1, epochs//5))
    train_time = time.time() - t0

    # Save weights in the format ref.py expects: W1..WL and b1..bL
    save_dict = {}
    L = len(model.weights)
    for i in range(L):
        save_dict[f"W{i+1}"] = model.weights[i]
        save_dict[f"b{i+1}"] = model.biases[i]
    # optional metadata (ref.py will ignore extra keys)
    save_dict["layer_sizes"] = np.array(model.layer_sizes, dtype=np.int64)
    save_dict["activation_func"] = model.activation_func_name
    save_dict["loss_func"] = model.loss_func_name
    np.savez("mymlp_weights.npz", **save_dict)
    print("Weights saved to mymlp_weights.npz")

    # Evaluate
    user_pred = model.predict(X_test)  # can be (C,m) or (m,)
    acc = accuracy_from_user_pred(user_pred, y_test_int)
    return train_time, acc


def run_torch_mlp(X_train, y_train_int, X_test, y_test_int, hidden, lr, epochs, seed):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    class TorchMLP(nn.Module):
        def __init__(self, hidden_list):
            super().__init__()
            layers = []
            in_dim = 784
            for h in hidden_list:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            layers += [nn.Linear(in_dim, 10)]
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    hidden_list = [int(h) for h in hidden.split(",") if h.strip()]
    net = TorchMLP(hidden_list).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    Xtr = torch.from_numpy(X_train.T.astype(np.float32)).to(device)
    ytr = torch.from_numpy(y_train_int.astype(np.int64)).to(device)
    Xte = torch.from_numpy(X_test.T.astype(np.float32)).to(device)
    yte = torch.from_numpy(y_test_int.astype(np.int64)).to(device)

    batch_size = 128
    n = Xtr.shape[0]
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]
            optimizer.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    train_time = time.time() - t0

    with torch.no_grad():
        logits = net(Xte)
        preds = logits.argmax(dim=1)
        acc = float((preds == yte).float().mean().item())
    return train_time, acc

def run_tf_mlp(X_train, y_train_int, X_test, y_test_int, hidden, lr, epochs, seed):
    try:
        import tensorflow as tf
    except Exception:
        return None, None

    tf.random.set_seed(seed)

    hidden_list = [int(h) for h in hidden.split(",") if h.strip()]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(784,)))
    for h in hidden_list:
        model.add(tf.keras.layers.Dense(h, activation='relu'))
    model.add(tf.keras.layers.Dense(10))  # logits
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    Xtr = X_train.T.astype(np.float32)
    ytr = y_train_int
    Xte = X_test.T.astype(np.float32)
    yte = y_test_int

    t0 = time.time()
    model.fit(Xtr, ytr, epochs=epochs, batch_size=128, verbose=0)
    train_time = time.time() - t0

    loss, acc = model.evaluate(Xte, yte, verbose=0)
    return train_time, float(acc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--subset', type=int, default=10000)
    parser.add_argument('--hidden', type=str, default='128,64')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("Loading MNIST (subset={} seed={})...".format(args.subset, args.seed))
    Xtr, Ytr_oh, ytr, Xte, Yte_oh, yte = load_mnist(subset=args.subset, seed=args.seed)
    print("Train:", Xtr.shape, Ytr_oh.shape, " Test:", Xte.shape, Yte_oh.shape)

    results = []

    print("\n[MLP] training... (full-batch)")
    try:
        t_user, acc_user = run_user_mlp(Xtr, Ytr_oh, Xte, yte, args.hidden, args.lr, args.epochs, args.seed)
        results.append(("MLP", t_user, acc_user))
        print(f"[MLP] time: {t_user:.2f}s  acc: {acc_user:.4f}")
    except Exception as e:
        print("[MLP] ERROR during training/eval:", repr(e))
    
    print("\n[PyTorch MLP] training...")
    t_torch, acc_torch = run_torch_mlp(Xtr, ytr, Xte, yte, args.hidden, args.lr, args.epochs, args.seed)
    if t_torch is not None:
        results.append(("PyTorch MLP", t_torch, acc_torch))
        print(f"[PyTorch] time: {t_torch:.2f}s  acc: {acc_torch:.4f}")
    else:
        print("[PyTorch] not available, skipping.")

    print("\n[TensorFlow MLP] training...")
    t_tf, acc_tf = run_tf_mlp(Xtr, ytr, Xte, yte, args.hidden, args.lr, args.epochs, args.seed)
    if t_tf is not None:
        results.append(("TensorFlow MLP", t_tf, acc_tf))
        print(f"[TensorFlow] time: {t_tf:.2f}s  acc: {acc_tf:.4f}")
    else:
        print("[TensorFlow] not available, skipping.")

    if results:
        print("\n=== Summary ===")
        name_w = max(len(n) for n,_,_ in results)
        for name, t, acc in results:
            print(f"{name:<{name_w}}  time: {t:>7.2f}s   acc: {acc:.4f}")
    else:
        print("No results to report. Did all frameworks fail to import?")

if __name__ == "__main__":
    main()