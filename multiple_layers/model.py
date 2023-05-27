import numpy as np
import matplotlib.pyplot as plt
from activations import *

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2 / input_size
        )
        self.biases = np.zeros(output_size)
        self.activation = activation
        self.dweights = None
        self.dbiases = None

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.activation.forward(self.output)

    def backward(self, dvalues):
        dinputs = self.activation.backward(dvalues)
        self.dweights = np.dot(self.input.T, dinputs)
        self.dbiases = np.sum(dinputs, axis=0)
        return np.dot(dinputs, self.weights.T)


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def configureTraining(self, epochs=1000, learning_rate=0.0001, batch_size=32, clip_threshold=5.0):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.clip_threshold = clip_threshold

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y, clip_threshold=None):
        dvalues = 2 * (self.layers[-1].output - y) / y.size
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
            if isinstance(layer, Layer):
                if clip_threshold is not None:
                    np.clip(
                        layer.dweights,
                        -clip_threshold,
                        clip_threshold,
                        out=layer.dweights,
                    )
                    np.clip(
                        layer.dbiases,
                        -clip_threshold,
                        clip_threshold,
                        out=layer.dbiases,
                    )
                layer.weights -= self.learning_rate * layer.dweights
                layer.biases -= self.learning_rate * layer.dbiases

    def train(self, x, y, num_samples):
        losses = []
        for self.epoch in range(self.epochs):
            epoch_loss = 0.0

            # Shuffle training data
            indices = np.random.permutation(num_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for idx in range(0, num_samples, self.batch_size):
                batch_x = x_shuffled[idx : idx + self.batch_size]
                batch_y = y_shuffled[idx : idx + self.batch_size]

                # Forward pass
                output = self.forward(batch_x)

                # Backward pass with gradient clipping
                self.backward(
                    batch_x,
                    batch_y.reshape(-1, 1),
                    clip_threshold=self.clip_threshold,
                )

                # Compute batch loss
                batch_loss = np.mean((output - batch_y.reshape(-1, 1)) ** 2)
                epoch_loss += batch_loss

            # Print progress
            loss = epoch_loss / (num_samples // self.batch_size)
            losses.append(loss)
            if self.epoch % 100 == 0:
                print(f"Epoch: {self.epoch}, Loss: {loss:.8f}")

            # Adjust learning rate (learning rate decay)
            if (self.epoch + 1) % 200 == 0:
                self.learning_rate *= 0.1

            # Check for NaN loss
            if np.isnan(loss):
                print("Loss became NaN. Terminating training.")
                break

        # Plot the loss curve
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    def test(self, x):
        predicted_sums = self.forward(x).flatten()

        # Visualize the results
        plt.scatter(x[:, 0], x[:, 1], c=predicted_sums)
        plt.colorbar(label="Predicted Sum")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Sum of Two Numbers")
        plt.show()

        for i, data in enumerate(x):
            print(f"{data} -> {predicted_sums[i]}")