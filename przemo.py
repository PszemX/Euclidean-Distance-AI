"""
COS TUTAJ PROBOWALEM KOMBINOWAC ALE CHUJ
"""

import numpy as np
from generators import Generator
from Activation2 import Activation_ReLU, Activation_Softmax, Activation_Sigmoid
import matplotlib.pyplot as plt

np.random.seed(0)
generator = Generator("80%")
x_train, y_train, x_test, y_test = generator.generate(
    size=100, min_range=0, max_range=10, type="addition"
)
# plt.scatter(x_train[:, 0], x_train[:, 1])
# plt.show()

# plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap="brg")
# plt.show()


class Neuron:
    inputs = None
    weights = None
    output = -np.Inf
    bias = -np.Inf

    def __init__(self, inputs, weights, bias):
        self.inputs = inputs
        self.weights = weights
        self.bias = bias
        self.output = np.dot(self.weights, self.inputs) + self.bias


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# Input
layer1 = Layer(n_inputs=2, n_neurons=3)
activation1 = Activation_Sigmoid()

layer2 = Layer(n_inputs=3, n_neurons=3)
activation2 = Activation_Sigmoid()

layer1.forward(x_train)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:3])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y_train)

print("Loss:", loss)
