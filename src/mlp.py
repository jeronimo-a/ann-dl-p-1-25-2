''' Class to define an MLP model '''

import numpy as np

from layer import Layer
from output_layer import OutputLayer

class MLP:

    rng: np.random.Generator
    seed: int

    hidden_layers: list[Layer]
    output_layer: OutputLayer
    n_inputs: int

    learning_rate: float

    trained_epochs: int = 0
    loss_history: list[float]

    def __init__(self, n_inputs: int, output_layer_config: dict, hidden_layers_config: list[dict] | None = None, learning_rate: float = 0.01, seed: int = None):

        # set up random number generator with constant seed for reproducibility
        self.seed = seed if seed is not None else np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(self.seed)

        # model parameters
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs

        # initializes output layer
        self.output_layer = OutputLayer(**output_layer_config, rng=self.rng)

        # initializes hidden layers
        self.hidden_layers = []
        if hidden_layers_config is None:
            hidden_layers_config = []
        for hidden_layer_config in hidden_layers_config:
            hidden_layer = Layer(**hidden_layer_config, rng=self.rng)
            self.hidden_layers.append(hidden_layer)
            hidden_layer.initialize_weights(n_inputs)
            n_inputs = hidden_layer.size
        self.output_layer.initialize_weights(n_inputs)

        self.loss_history = []

    def forward(self, X: np.ndarray) -> np.ndarray:

        # input validation
        if X.ndim != 2 or X.shape != (self.n_inputs, 1): raise ValueError(f"Input shape {X.shape} does not match expected shape {(self.n_inputs, 1)}.")

        a = X
        for hidden_layer in self.hidden_layers:
            a = hidden_layer.forward(a)
        a = self.output_layer.forward(a)
        return a
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        dx = self.output_layer.backward(y_true, y_pred)
        for hidden_layer in reversed(self.hidden_layers):
            dx = hidden_layer.backward(dx)

    def update_parameters(self):
        for hidden_layer in self.hidden_layers:
            hidden_layer.update_parameters(self.learning_rate)
        self.output_layer.update_parameters(self.learning_rate)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1):

        # input validation
        if X.ndim != 2: raise ValueError(f"Input data X must be a 2D array, but got {X.ndim}D array.")
        if y.ndim != 2: raise ValueError(f"Output data y must be a 2D array, but got {y.ndim}D array.")
        if X.shape[1] != self.n_inputs: raise ValueError(f"Input data has {X.shape[1]} features, but model expects {self.n_inputs} features.")
        if X.shape[0] != y.shape[0]: raise ValueError(f"Number of samples in X ({X.shape[1]}) does not match number of samples in y ({y.shape[1]}).")
        if y.shape[1] != self.output_layer.size: raise ValueError(f"Output data has {y.shape[1]} features, but model expects {self.output_layer.size} features.")
        
        for _ in range(epochs):
            epoch_loss = 0
            for i in range(X.shape[0]):
                x_i = X[i,np.newaxis].T
                y_i = y[i,np.newaxis].T
                y_pred = self.forward(x_i)
                epoch_loss += self.output_layer.loss(y_i, y_pred)
                self.backward(y_i, y_pred)
                self.update_parameters()
            self.loss_history.append(epoch_loss / X.shape[1])
            self.trained_epochs += 1
            print(f"Epoch {self.trained_epochs}/{epochs}, Loss: {epoch_loss:.4f}")

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        
        # validate inputs
        if X.ndim != 2: raise ValueError(f"Input data X must be a 2D array, but got {X.ndim}D array.")
        if y.ndim != 2: raise ValueError(f"Output data y must be a 2D array, but got {y.ndim}D array.")
        if X.shape[1] != self.n_inputs: raise ValueError(f"Input data has {X.shape[1]} features, but model expects {self.n_inputs} features.")
        if X.shape[0] != y.shape[0]: raise ValueError(f"Number of samples in X ({X.shape[1]}) does not match number of samples in y ({y.shape[1]}).")
        if y.shape[1] != self.output_layer.size: raise ValueError(f"Output data has {y.shape[1]} features, but model expects {self.output_layer.size} features.")

        n_correct = 0
        n_total = X.shape[0]

        for i in range(n_total):
            y_pred = self.forward(X[i,np.newaxis].T)
            y_actual = y[i,np.newaxis].T
            if self.output_layer.size > 1:
                y_pred = np.argmax(y_pred.flatten())
                y_actual = np.argmax(y_actual.flatten())
            elif self.output_layer.size == 1:
                y_pred = round(y_pred[0,0])
                y_actual = round(y_actual[0,0])

            n_correct += int(y_pred == y_actual)

        return n_correct / n_total
    
    def save(self, filepath: str):
        layer_params = dict()
        layer_params["n_inputs"] = self.n_inputs
        for i, hidden_layer in enumerate(self.hidden_layers):
            layer_params[f"hidden_layer_{i}"] = {
                "size": hidden_layer.size,
                "activation_function": hidden_layer.activation_function,
                "weights_initializer": hidden_layer.weights_initializer,
                "initial_biases": hidden_layer.initial_biases,
                "weights": hidden_layer.weights,
                "biases": hidden_layer.biases
            }
        layer_params["output_layer"] = {
            "size": self.output_layer.size,
            "activation_function": self.output_layer.activation_function,
            "weights_initializer": self.output_layer.weights_initializer,
            "initial_biases": self.output_layer.initial_biases,
            "loss_function": self.output_layer.loss_function,
            "weights": self.output_layer.weights,
            "biases": self.output_layer.biases
        }
        layer_params["learning_rate"] = self.learning_rate
        layer_params["seed"] = self.seed
        layer_params["trained_epochs"] = self.trained_epochs
        np.savez_compressed(filepath, **layer_params)