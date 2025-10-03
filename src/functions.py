'''
Holds activation functions, loss functions, and their derivatives.
As well as weight initialization methods.
'''

import numpy as np

EPSILON = 1e-15

# -- WEIGTH INITIALIZERS -- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def xavier_uniform(n_input: int, n_output: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6 / (n_input + n_output))
    return rng.uniform(-limit, limit, (n_output, n_input))

def xavier_normal(n_input: int, n_output: int, rng: np.random.Generator) -> np.ndarray:
    stddev = np.sqrt(2 / (n_input + n_output))
    return rng.normal(0, stddev, (n_output, n_input))

def he_uniform(n_input: int, n_output: int, rng: np.random.Generator) -> np.ndarray:
    limit = np.sqrt(6 / n_input)
    return rng.uniform(-limit, limit, (n_output, n_input))

def he_normal(n_input: int, n_output: int, rng: np.random.Generator) -> np.ndarray:
    stddev = np.sqrt(2 / n_input)
    return rng.normal(0, stddev, (n_output, n_input))

# -- ACTIVATION FUNCTIONS - --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x)**2

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / e_x.sum(axis=0, keepdims=True)

# -- LOSS FUNCTIONS --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = y_true.shape[0]
    EPSILON = 1e-15
    y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
    loss = - (1 / m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    EPSILON = 1e-15
    y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
    loss = - np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def bce_wrt_sigmoid_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    EPSILON = 1e-15
    y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
    return - (y_true / y_pred) + ((1 - y_true) / (1 - y_pred))

def cce_wrt_softmax_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_pred - y_true