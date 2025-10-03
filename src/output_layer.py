'''  '''

import numpy as np
import functions as fn

from layer import Layer

class OutputLayer(Layer):

    loss_function: str

    def __init__(self, size: int, loss_function: str, activation_function: str, weights_initializer: str, initial_biases: int = 0, seed: int = None):
        super().__init__(size, activation_function, weights_initializer, initial_biases, seed)
        self.loss_function = loss_function.lower()

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        match self.loss_function:
            case "bce": return fn.binary_cross_entropy(y_true, y_pred)
            case "cce": return fn.categorical_cross_entropy(y_true, y_pred)
            case _: raise ValueError(f"Unknown loss function: {self.loss_function}. Must be one of 'bce' or 'cce'.")

    def loss_derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        match (self.loss_function, self.activation_function):
            case ("bce", "sigmoid"): return fn.bce_wrt_sigmoid_derivative(y_true, y_pred)
            case _: raise ValueError(f"Loss derivative not implemented for loss function '{self.loss_function}' with activation function '{self.activation_function}'.")

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        m = y_true.shape[1]

        if self.activation_function == "softmax" and self.loss_function == "cce":
            dz = y_pred - y_true
        else:
            dA = self.loss_derivative(y_true, y_pred)
            dz = dA * self.activation_derivative(self.cache['z'])

        self.cache['dW'] = (1 / m) * (dz @ self.cache['a_prev'].T)
        self.cache['db'] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dX = self.weights.T @ dz
        return dX