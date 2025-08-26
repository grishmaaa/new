import numpy as np
class ActivationFunction:
    
    
    @staticmethod
    def sigmoid(z):
        """Computes the sigmoid of z."""
        # Clip z to avoid overflow in exp
        z = np.clip(z, -500, 500)                                       ############################################################################
        return 1 / (1 + np.exp(-z))                                     ############################################################################

    @staticmethod
    def sigmoid_derivative(z):
        """Computes the derivative of the sigmoid function."""
        s = ActivationFunction.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def relu(z):
        """Computes the Rectified Linear Unit of z."""
        return np.maximum(0, z)                                    ############################################################################

    @staticmethod
    def relu_derivative(z):
        """Computes the derivative of the ReLU function."""
        # The derivative is 1 for z > 0, and 0 otherwise.
        dZ = np.array(z, copy=True)                                           ############################################################################
        dZ[dZ <= 0] = 0
        dZ[dZ > 0] = 1
        return dZ
        
    @staticmethod
    def tanh(z):
        """Computes the hyperbolic tangent of z."""
        return np.tanh(z)                                     ############################################################################

    @staticmethod
    def tanh_derivative(z):
        """Computes the derivative of the tanh function."""
        return 1 - np.power(ActivationFunction.tanh(z), 2)

    @staticmethod
    def softmax(z):
        """Computes the softmax of z."""
        # Subtracting the max of z for numerical stability (prevents overflow).
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))                    ############################################################################
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    @staticmethod
    def softmax_derivative(z):  
        """Computes the derivative of the softmax function."""
        s = ActivationFunction.softmax(z)
        return s * (1 - s)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        """Computes the Leaky ReLU of z."""
        return np.where(z > 0, z, alpha * z)                                     ############################################################################

    @staticmethod
    def leaky_relu_derivative(z, alpha=0.01):
        """Computes the derivative of the Leaky ReLU function."""
        dz = np.ones_like(z)                                                     ############################################################################
        dz[z < 0] = alpha
        return dz

    @staticmethod
    def elu(z, alpha=1.0):
        """Computes the Exponential Linear Unit of z.""" 
        return np.where(z >= 0, z, alpha * (np.exp(z) - 1))                      ############################################################################

    @staticmethod
    def elu_derivative(z, alpha=1.0):
        """Computes the derivative of the ELU function."""
        return np.where(z >= 0, 1, alpha * np.exp(z))                            ############################################################################

    @staticmethod
    def swish(z, beta=1.0):
        """Computes the Swish activation function."""
        return z * ActivationFunction.sigmoid(beta * z)

    @staticmethod
    def swish_derivative(z, beta=1.0):
        """Computes the derivative of the Swish function."""
        sig = ActivationFunction.sigmoid(beta * z)
        return sig + beta * z * sig * (1 - sig)

    @staticmethod
    def softplus(z):
        """Computes the Softplus activation function."""
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)                      ############################################################################

    @staticmethod
    def softplus_derivative(z):
        """Computes the derivative of the Softplus function."""
        return ActivationFunction.sigmoid(z)
    
    @staticmethod
    def identity(z):
        """Identity activation function."""
        return z

    @staticmethod
    def identity_derivative(z):
        """Derivative of the identity function."""
        return np.ones_like(z)                                                ############################################################################
    
    @staticmethod
    def get(name: str):
        table = {
            'relu': (ActivationFunction.relu, ActivationFunction.relu_derivative),
            'sigmoid': (ActivationFunction.sigmoid, ActivationFunction.sigmoid_derivative),
            'tanh': (ActivationFunction.tanh, ActivationFunction.tanh_derivative),
            'leaky_relu': (ActivationFunction.leaky_relu, ActivationFunction.leaky_relu_derivative),
            'softmax': (ActivationFunction.softmax, ActivationFunction.softmax_derivative),
            'elu': (ActivationFunction.elu, ActivationFunction.elu_derivative),
            'swish': (ActivationFunction.swish, ActivationFunction.swish_derivative),
            'softplus': (ActivationFunction.softplus, ActivationFunction.softplus_derivative),
            'identity': (ActivationFunction.identity, ActivationFunction.identity_derivative),
        }
        return table[name]

    @staticmethod
    def output_for_loss(loss_name: str):
        mapping = {
            'binary_cross_entropy_loss': 'sigmoid',
            'categorical_cross_entropy_loss': 'softmax',
            'cross_entropy_loss': 'softmax',
            'mean_squared_error': 'identity',
            'mean_absolute_error': 'identity',
            'hinge_loss': 'identity',
        }
        return ActivationFunction.get(mapping[loss_name])