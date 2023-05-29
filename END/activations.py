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
    
class ActivationSigmoid:
    def forward(self, input):
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, dvalues):
        sigmoid = 1 / (1 + np.exp(-self.input))
        self.dinputs = dvalues * sigmoid * (1 - sigmoid)
        return self.dinputs


class ActivationTanh:
    def forward(self, input):
        self.input = input
        self.output = np.tanh(input)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - np.tanh(self.input) ** 2)
        return self.dinputs