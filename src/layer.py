''' Defines the layer class, which will compose the neural network '''

import numpy as np

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
            
            # xavier cases
            case "xavier-uniform":
                limit = np.sqrt(6 / (n_input + self.size))
                self.weights = self.rng.uniform(-limit, limit, (self.size, n_input))
            case "xavier-normal":
                stddev = np.sqrt(2 / (n_input + self.size))
                self.weights = self.rng.normal(0, stddev, (self.size, n_input))

            # he cases
            case "he-uniform":
                limit = np.sqrt(6 / n_input)
                self.weights = self.rng.uniform(-limit, limit, (self.size, n_input))
            case "he-normal":
                stddev = np.sqrt(2 / n_input)
                self.weights = self.rng.normal(0, stddev, (self.size, n_input))

            # unknown initializer
            case _:
                raise ValueError(f"Unknown weight initializer: {self.weight_initializer}. Must be one of 'xavier-uniform', 'xavier-normal', 'he-uniform', or 'he-normal'.")
    
    