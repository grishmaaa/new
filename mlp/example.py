import numpy as np
from mlp import MLP


np.random.seed(42)
x = np.random.randn(10, 10)

mlp = MLP(10,[4], 1, activation_func='relu', loss_function='mean_squared_error', learning_rate=0.01)
output = mlp.forward(x)
# update = mlp.train(x, output, np.random.randn(1, 10))
print(output)
# print(update)