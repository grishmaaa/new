import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from mlp import MLP

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
y = y.astype(np.float64).reshape(1, -1)  # ensure float for BCE

X_train, X_test, y_train, y_test = train_test_split(X, y.T, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = X_train.T, X_test.T, y_train.T, y_test.T

print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test data shape:     X={X_test.shape}, y={y_test.shape}")

mlp = MLP(
    input_size=2,
    hidden_layers=[4,4],
    output_size=1,
    weight_initialization='he_normal',
    activation_func='relu',
    loss_function='binary_cross_entropy_loss',  # <-- important
    learning_rate=0.05
)

costs = mlp.train(X_train, y_train, epochs=2500, print_cost_every=250)
pred = mlp.predict(X_test)
acc = np.mean(pred == y_test) * 100.0
print(f"Test Accuracy: {acc:.2f}%")
