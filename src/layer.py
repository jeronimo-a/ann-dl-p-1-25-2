''' Defines the layer class, which will compose the neural network '''

import numpy as np
import functions as fn

class Layer:

    rng: np.random.Generator

    size: int
    n_input: int

    activation_function: str
    weights_initializer: str

    initial_biases: int
    weights: np.ndarray
    biases: np.ndarray

    cache: dict

    def __init__(self, size: int, activation_function: str, weights_initializer: str, rng: np.random.Generator, initial_biases: int = 0):

        # attaches random number generator
        self.rng = rng

        # layer parameters
        self.size = size
        self.activation_function = activation_function.lower()
        
        # parameter initialization parameters
        self.weights_initializer = weights_initializer.lower()
        self.initial_biases = initial_biases

        # initializes cache
        self.cache = {}

    def initialize_weights(self, n_input: int):

        self.n_input = n_input

        # biases initialization
        self.biases = np.full((self.size, 1), self.initial_biases, dtype=np.float64)

        # weights initialization
        match self.weights_initializer:
            case "xavier-uniform": self.weights = fn.xavier_uniform(self.n_input, self.size, self.rng)
            case "xavier-normal": self.weights = fn.xavier_normal(self.n_input, self.size, self.rng)
            case "he-uniform": self.weights = fn.he_uniform(self.n_input, self.size, self.rng)
            case "he-normal": self.weights = fn.he_normal(self.n_input, self.size, self.rng)
            case _: raise ValueError(f"Unknown weight initializer: {self.weight_initializer}. Must be one of 'xavier-uniform', 'xavier-normal', 'he-uniform', or 'he-normal'.")
            
    def activation(self, x: np.ndarray) -> np.ndarray:
        match self.activation_function:
            case "sigmoid": return fn.sigmoid(x)
            case "tanh": return fn.tanh(x)
            case "relu": return fn.relu(x)
            case "softmax": return fn.softmax(x)
            case _: raise ValueError(f"Unknown activation function: {self.activation_function}. Must be one of 'sigmoid', 'tanh', 'relu' or 'softmax'.")

    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        match self.activation_function:
            case "sigmoid": return fn.sigmoid_derivative(x)
            case "tanh": return fn.tanh_derivative(x)
            case "relu": return fn.relu_derivative(x)
            case _: raise ValueError(f"Unknown activation function: {self.activation_function}. Must be one of 'sigmoid', 'tanh', or 'relu'.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x'] = x
        self.cache['z'] = self.weights @ x + self.biases
        self.cache['a'] = self.activation(self.cache['z'])
        return self.cache['a']

    def backward(self, da: np.ndarray) -> np.ndarray:
        delta = da * self.activation_derivative(self.cache['z'])
        self.cache['dW'] = (delta @ self.cache['x'].T)
        self.cache['db'] = delta
        dX = self.weights.T @ delta
        return dX
    
    def update_parameters(self, learning_rate: float):
        self.weights -= learning_rate * self.cache['dW']
        self.biases -= learning_rate * self.cache['db']