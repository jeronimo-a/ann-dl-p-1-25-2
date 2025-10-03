''' Defines the layer class, which will compose the neural network '''

import numpy as np
import functions as fn

class Layer:

    rng: np.random.Generator
    seed: int

    size: int
    loss_function: str
    activation_function: str

    weights_initializer: str
    initial_weights: np.ndarray
    initial_biases: int

    weights: np.ndarray
    biases: np.ndarray

    def __init__(self, size: int, loss_function: str, activation_function: str, weights_initializer: str, initial_biases: int = 0, seed: int = None):

        # set up random number generator with constant seed for reproducibility
        self.seed = seed if seed is not None else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(self.seed)

        # layer parameters
        self.size = size
        self.loss_function = loss_function.lower()
        self.activation_function = activation_function.lower()
        
        # parameter initialization parameters
        self.weights_initializer = weights_initializer.lower()
        self.initial_biases = initial_biases

    def initialize_weights(self, n_input: int):

        # biases initialization
        self.biases = np.full((self.size, 1), self.initial_biases)

        # weights initialization
        match self.weights_initializer:
            case "xavier-uniform": self.weights = fn.xavier_uniform(n_input, self.size, self.rng)
            case "xavier-normal": self.weights = fn.xavier_normal(n_input, self.size, self.rng)
            case "he-uniform": self.weights = fn.he_uniform(n_input, self.size, self.rng)
            case "he-normal": self.weights = fn.he_normal(n_input, self.size, self.rng)
            case _: raise ValueError(f"Unknown weight initializer: {self.weight_initializer}. Must be one of 'xavier-uniform', 'xavier-normal', 'he-uniform', or 'he-normal'.")
            
    def activation(self, x: np.ndarray) -> np.ndarray:
        match self.activation_function:
            case "sigmoid": return fn.sigmoid(x)
            case "tanh": return fn.tanh(x)
            case "relu": return fn.relu(x)
            case _: raise ValueError(f"Unknown activation function: {self.activation_function}. Must be one of 'sigmoid', 'tanh', or 'relu'.")

    def activation_derivative(self, x: np.ndarray) -> np.ndarray:
        match self.activation_function:
            case "sigmoid": return fn.sigmoid_derivative(x)
            case "tanh": return fn.tanh_derivative(x)
            case "relu": return fn.relu_derivative(x)
            case _: raise ValueError(f"Unknown activation function: {self.activation_function}. Must be one of 'sigmoid', 'tanh', or 'relu'.")

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        match self.loss_function:
            case "binary-cross-entropy": return fn.binary_cross_entropy(y_true, y_pred)
            case "categorical-cross-entropy": return fn.categorical_cross_entropy(y_true, y_pred)
            case _: raise ValueError(f"Unknown loss function: {self.loss_function}. Must be one of 'bce' or 'cce'.")