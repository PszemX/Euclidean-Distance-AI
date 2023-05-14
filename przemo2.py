import numpy as np
from generators import Generator


class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights1 = (
            2 * np.random.random((2, 4)) - 1
        )  # Dodatkowa warstwa ukryta
        self.synaptic_weights2 = 2 * np.random.random((4, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, num_iterations):
        for iteration in range(num_iterations):
            output1, output2 = self.think(training_inputs)

            error2 = training_outputs - output2
            delta2 = error2 * self.sigmoid_derivative(output2)

            error1 = delta2.dot(self.synaptic_weights2.T)
            delta1 = error1 * self.sigmoid_derivative(output1)

            adjustment2 = output1.T.dot(delta2)
            adjustment1 = training_inputs.T.dot(delta1)

            self.synaptic_weights2 += adjustment2
            self.synaptic_weights1 += adjustment1

    def think(self, inputs):
        output1 = self.sigmoid(np.dot(inputs, self.synaptic_weights1))
        output2 = self.sigmoid(np.dot(output1, self.synaptic_weights2))
        return output1, output2


training_inputs = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
training_outputs = np.array([[0, 1, 2, 1, 2, 3]]).T

neural_network = NeuralNetwork()
neural_network.train(training_inputs, training_outputs, 500000)

print("Wagi po nauce:")
print("Warstwa ukryta:")
print(neural_network.synaptic_weights1)
print("Warstwa wyjściowa:")
print(neural_network.synaptic_weights2)

print("0 + 0 =", neural_network.think(np.array([0, 0]))[1])
print("0 + 1 =", neural_network.think(np.array([0, 1]))[1])
print("1 + 0 =", neural_network.think(np.array([1, 0]))[1])
print("1 + 1 =", neural_network.think(np.array([1, 1]))[1])
