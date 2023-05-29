import numpy as np
learning_rate = 0.001

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.001):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.t = 0

    def update_optimizer(self, learning_rise_value):
        self.learning_rate *= learning_rise_value

    def update_layer(self, layer):
        self.t += 1

        # Weights
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.dweights
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * (layer.dweights**2)

        m_hat = layer.m_weights / (1 - self.beta1**self.t)
        v_hat = layer.v_weights / (1 - self.beta2**self.t)

        layer.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Biases
        layer.m_biases = self.beta1 * layer.m_biases + (1 - self.beta1) * layer.dbiases
        layer.v_biases = self.beta2 * layer.v_biases + (1 - self.beta2) * (layer.dbiases**2)

        m_hatb = layer.m_biases / (1 - self.beta1**self.t)
        v_hatb = layer.v_biases / (1 - self.beta2**self.t)

        layer.biases -= self.learning_rate * m_hatb / (np.sqrt(v_hatb) + self.epsilon)