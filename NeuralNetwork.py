import numpy as np
import math

class DenseLayer:

    def __init__(self, n_input, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # ReLU activation function for values < 0 return 0, otherwise return value
    def activation_ReLU(self, X):
        return np.maximum(0, X)
    
    # Linear activation function for values return values, it is also named non activation function
    def adctivation_linear(self, X):
        return np.ones_like(X)
        
    # Sigmoid activation function return count from pattern, return between 0 and 1
    def adctivation_sigmoid(self, X):
        return X/(1 - ((math.e)**X))

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        # Simply output input * weight + bias
        self.output = self.activation_sigmoid(np.dot(inputs, self.weights) + self.biases)