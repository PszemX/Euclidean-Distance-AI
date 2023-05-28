import numpy as np
import matplotlib.pyplot as plt
from activations import *
from optimizers import *


class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2 / input_size
        )
        self.biases = np.zeros(output_size)
        self.activation = activation


        ################################################################
        self.dweights = None
        self.dbiases = None
        self.vw = np.zeros_like(self.weights)
        self.vb = np.zeros_like(self.biases)
        self.sw = np.zeros_like(self.weights)
        self.sb = np.zeros_like(self.biases)

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
        self.loss = 0

    def add(self, layer):
        self.layers.append(layer)

    def configureTraining(
        self,
        epochs=1000,
        batch_size=None,
        clip_threshold=None,
        optimizer=Optimizer_Adam(),
        accuracy=0.1,
        learning_rate_change_frequency = 10,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_threshold = clip_threshold
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.learning_rate_change_frequency = learning_rate_change_frequency

    def countAccuracy(self, predicted, y_real):
        return (
            np.mean(
                np.where(
                    ((predicted + self.accuracy) >= y_real) & ((predicted - self.accuracy) <= y_real), 1, 0
                )
            )
            * 100
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y):
        dvalues = 2 * (self.layers[-1].output - y) / y.size
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
            if self.clip_threshold is not None:
                np.clip(
                    layer.dweights,
                    -self.clip_threshold,
                    self.clip_threshold,
                    out=layer.dweights,
                )
                np.clip(
                    layer.dbiases,
                    -self.clip_threshold,
                    self.clip_threshold,
                    out=layer.dbiases,
                )

            # Update parameters by optimizer
            self.optimizer.update_layer(layer)

    def train(self, x, y):
        losses = []

        if self.batch_size is None:
            self.batch_size = len(x)

        for self.epoch in range(self.epochs):
            epoch_loss = 0.0

            # Shuffle training data
            indices = np.random.permutation(len(x))
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            for idx in range(0, len(x), self.batch_size):
                batch_x = x_shuffled[idx : idx + self.batch_size]
                batch_y = y_shuffled[idx : idx + self.batch_size]

                # Forward pass
                output = self.forward(batch_x)

                # Backward pass with gradient clipping
                self.backward(batch_x, batch_y.reshape(-1, 1))
                
                # Compute batch loss
                batch_loss = np.mean((output - batch_y.reshape(-1, 1)) ** 2)
                epoch_loss += batch_loss

            # Print progress
            loss = epoch_loss / (len(x) // self.batch_size)
            losses.append(loss)
            if self.epoch % self.learning_rate_change_frequency == 0:
                print(f"Epoch: {self.epoch}, Loss: {loss:.8f}")

            # Adjust learning rate (learning rate decay)
            if (self.epoch + 1) % 10 == 0:
                self.optimizer.pre_update_params()
                self.optimizer.post_update_params()

            # Check for NaN loss
            if np.isnan(loss):
                print("Loss became NaN. Terminating training.")
                break
        self.loss = loss
        # Plot the loss curve
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()

    def test(self, x, y):
        predicted = self.forward(x).flatten()

        # Calculate accuracy
        accuracy = self.countAccuracy(predicted=predicted, y_real=y)

        # Visualize the results
        plt.scatter(x[:, 0], x[:, 1], c=predicted)
        plt.colorbar(label="Predicted Sum")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Sum of Two Numbers")
        plt.show()

        for i, data in enumerate(x):
            print(f"{data} -> {predicted[i]}")

        print("\nAccuracy:")
        print(f"{accuracy:.2f} %")
