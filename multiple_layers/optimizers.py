import numpy as np

# Adam optimizer
class Optimizer_Adam:
# Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-8, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = self.learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0
        self.t = 0

    # Call once before any parameter updates
    def pre_update_params(self):
        self.learning_rate *= 0.1
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))

    def update_layer(self, layer):
        self.t += 1
        layer.vw = self.beta1 * layer.vw + (1 - self.beta1) * layer.dweights
        layer.vb = self.beta1 * layer.vb + (1 - self.beta1) * layer.dbiases

        layer.sw = self.beta2 * layer.sw + (1 - self.beta2) * (layer.dweights**2)
        layer.sb = self.beta2 * layer.sb + (1 - self.beta2) * (layer.dbiases**2)

        vw_corrected = layer.vw / (1 - self.beta1**self.t)
        vb_corrected = layer.vb / (1 - self.beta1**self.t)

        sw_corrected = layer.sw / (1 - self.beta2**self.t)
        sb_corrected = layer.sb / (1 - self.beta2**self.t)

        layer.weights -= (
            self.current_learning_rate
            * vw_corrected
            / (np.sqrt(sw_corrected) + self.epsilon)
        )

        layer.biases -= (
            self.current_learning_rate
            * vb_corrected
            / (np.sqrt(sb_corrected) + self.epsilon)
        )    

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1