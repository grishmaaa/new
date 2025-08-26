import numpy as np
import time
from ActivationFunctions import ActivationFunction
from WeightInitialization import WeightInitializer
from LossFunctions import LossFunction

class MLP:
    def __init__(self, input_size, hidden_layers, output_size, 
                 weight_initialization='he_normal', activation_func='relu', 
                 loss_function='binary_cross_entropy_loss', learning_rate=0.01):

        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layer_sizes)
        self.learning_rate = learning_rate
        self.weight_init_method = weight_initialization
        self.activation_func_name = activation_func
        self.loss_func_name = loss_function

        # Resolve everything from the libraries
        self.hidden_activation, self.hidden_activation_derivative = ActivationFunction.get(self.activation_func_name)
        self.output_activation, self.output_activation_derivative = ActivationFunction.output_for_loss(self.loss_func_name)
        self.cost, self.cost_derivative = LossFunction.get(self.loss_func_name)
        self.initializer = WeightInitializer.get(self.weight_init_method)

        self.weights = []
        self.biases = []
        self._initialize_parameters()

        self.cache = {}
        self.grads = {}

        print("MLP initialized successfully.")
        print(f"  - Architecture: {self.layer_sizes}")
        print(f"  - Hidden Activation: {self.activation_func_name}")
        print(f"  - Output Activation (from loss): {self.output_activation.__name__}")
        print(f"  - Weight Initialization: {self.weight_init_method}")
        print(f"  - Loss Function: {self.loss_func_name}")

    def _initialize_parameters(self):
        for i in range(1, self.num_layers):
            in_dim, out_dim = self.layer_sizes[i-1], self.layer_sizes[i]
            # Expect initializer(in_dim, out_dim) -> W of shape (out_dim, in_dim)
            W = self.initializer(in_dim, out_dim)
            b = np.zeros((out_dim, 1), dtype=np.float64)  #########################################################################
            self.weights.append(W.astype(np.float64))
            self.biases.append(b)

    def forward(self, X):
        self.cache = {'A0': X}
        A = X
        for l in range(1, self.num_layers):
            W, b = self.weights[l-1], self.biases[l-1]
            Z = np.dot(W, A) + b                           ############################################################################
            act = self.hidden_activation if l < self.num_layers - 1 else self.output_activation
            A = act(Z)
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A
        return A

    def backward(self, Y):
        # Shapes: Y: (out_dim, m), AL: (out_dim, m)
        m = Y.shape[1]
        L = self.num_layers - 1
        AL = self.cache[f'A{L}']

        # Stable gradient shortcuts for common pairs
        uses_softmax_ce = (self.output_activation.__name__ == 'softmax' and 
                           self.loss_func_name in ('categorical_cross_entropy_loss', 'cross_entropy_loss'))
        uses_sigmoid_bce = (self.output_activation.__name__ == 'sigmoid' and 
                            self.loss_func_name == 'binary_cross_entropy_loss')

        if uses_softmax_ce or uses_sigmoid_bce:
            dZ = AL - Y
        else:
            # generic path
            dAL = self.cost_derivative(Y, AL) if self.cost_derivative is not None else (AL - Y)
            dZ = dAL * self.output_activation_derivative(self.cache[f'Z{L}'])

        A_prev = self.cache[f'A{L-1}']
        self.grads[f'dW{L}'] = (1.0/m) * np.dot(dZ, A_prev.T)                                ############################################################################
        self.grads[f'db{L}'] = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)                    ############################################################################
        dA = np.dot(self.weights[L-1].T, dZ)                                                 ############################################################################

        for l in reversed(range(1, L)):
            dZ = dA * self.hidden_activation_derivative(self.cache[f'Z{l}'])
            A_prev = self.cache[f'A{l-1}']
            self.grads[f'dW{l}'] = (1.0/m) * np.dot(dZ, A_prev.T)                             ############################################################################
            self.grads[f'db{l}'] = (1.0/m) * np.sum(dZ, axis=1, keepdims=True)                 ############################################################################
            dA = np.dot(self.weights[l-1].T, dZ)                                              ############################################################################

    def update_parameters(self):
        for l in range(1, self.num_layers):
            self.weights[l-1] -= self.learning_rate * self.grads[f'dW{l}']
            self.biases[l-1]  -= self.learning_rate * self.grads[f'db{l}']

    def train(self, X, Y, epochs, print_cost_every=100):
        print(f"\n--- Starting Training for {epochs} epochs ---")
        start_time = time.time()
        costs = []

        for i in range(epochs):
            AL = self.forward(X)
            self.backward(Y)
            self.update_parameters()

            if i % print_cost_every == 0 or i == epochs - 1:
                cost = float(self.cost(Y, AL))
                costs.append(cost)
                print(f"Epoch {i: >4} | Cost: {cost:.6f}")

        dt = time.time() - start_time
        print(f"--- Training finished in {dt:.2f} seconds ---")
        return costs

    def predict(self, X):
        AL = self.forward(X)
        # Binary case -> threshold
        if self.loss_func_name == 'binary_cross_entropy_loss' and self.layer_sizes[-1] == 1:
            return (AL > 0.5).astype(int)
        # Multiclass -> argmax
        if self.loss_func_name in ('categorical_cross_entropy_loss', 'cross_entropy_loss'):
            return np.argmax(AL, axis=0)                                                                   ############################################################################
        # Fallback: return raw activations
        return AL