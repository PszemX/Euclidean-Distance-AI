import numpy as np

class ActivationLeakyReLU:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(input, 0.01 * input)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] *= 0.01
        return self.dinputs


class ActivationReLU:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
        return self.dinputs