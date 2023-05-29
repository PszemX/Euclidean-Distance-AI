import numpy as np
learning_rate = 0.001

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.001):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.t = 0

    def update(self, weights, gradients, m, v):
        self.t += 1

        m = self.beta1 * m + (1 - self.beta1) * gradients
        v = self.beta2 * v + (1 - self.beta2) * (gradients**2)

        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)

        weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights, m, v
