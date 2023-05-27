import numpy as np

# Adam optimizer
class Optimizer_Adam:
# Initialize optimizer - set settings
    def __init__(self, learning_rate=0.001,  epsilon=1e-8, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update_layer(self, layer):
        layer.vw = self.beta1 * layer.vw + (1 - self.beta1) * layer.dweights
        layer.vb = self.beta1 * layer.vb + (1 - self.beta1) * layer.dbiases

        layer.sw = self.beta2 * layer.sw + (1 - self.beta2) * (layer.dweights**2)
        layer.sb = self.beta2 * layer.sb + (1 - self.beta2) * (layer.dbiases**2)

        vw_corrected = layer.vw / (1 - self.beta1)
        vb_corrected = layer.vb / (1 - self.beta1)

        sw_corrected = layer.sw / (1 - self.beta2)
        sb_corrected = layer.sb / (1 - self.beta2)

        layer.weights -= (
            self.learning_rate
            * vw_corrected
            / (np.sqrt(sw_corrected) + self.epsilon)
        )

        layer.biases -= (
            self.learning_rate
            * vb_corrected
            / (np.sqrt(sb_corrected) + self.epsilon)
        )    