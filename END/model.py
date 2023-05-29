import numpy as np
import matplotlib.pyplot as plt
from optimizers import *
from generators import *
from activations import *

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * (
            2 / (input_size + output_size)
        )
        self.biases = np.zeros(output_size)
        self.activation = activation
        self.dweights = None
        self.dbiases = None
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.activation.forward(self.output)

    def backward_prepare(self, dvalues):
        self.dinputs = self.activation.backward(dvalues)
        self.dweights = np.dot(self.input.T, self.dinputs)
        self.dbiases = np.sum(self.dinputs, axis=0)
    
    def backward_update(self):
        return np.dot(self.dinputs, self.weights.T)


class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def addLayer(self, layer):
        self.layers.append(layer)

    def configure(
        self,
        epochs=1000,
        batch_size=None,
        clip_threshold=5.0,
        optimizer=None,
        learning_rate=0.001,
        learning_rise_interval = 200,
        learning_rise_value = 0.1
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.clip_threshold = clip_threshold
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.learning_rise_interval = learning_rise_interval
        self.learning_rise_value = learning_rise_value

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y):
        dvalues = 2 * (self.layers[-1].output - y) / y.size
        for layer in reversed(self.layers):
            layer.backward_prepare(dvalues)
            if self.optimizer:
                self.optimizer.update_layer(layer=layer)
            else:
                self.weights -= learning_rate * self.dweights
                self.biases -= learning_rate * self.dbiases
            dvalues = layer.backward_update()
            if isinstance(layer, Layer):
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


    def train(self, x, y):
        losses = []
        learning_rate = 0.001
        for epoch in range(self.epochs):
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

                # Backward pass with gradient clipping and parameter optimization
                self.backward(batch_y.reshape(-1, 1))

                # Compute batch loss
                batch_loss = np.mean((output - batch_y.reshape(-1, 1)) ** 2)
                epoch_loss += batch_loss

            # Print progress
            loss = epoch_loss / (len(x) // self.batch_size)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss:.8f}")

            # Adjust learning rate (learning rate decay)
            if (epoch + 1) % self.learning_rise_interval == 0:
                if self.optimizer is not None:
                    self.optimizer.update_optimizer(self.learning_rise_value)
                else: 
                    learning_rate *= 0.1

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

    def test(self, x, y):
        # Generate test data
        test_x1, test_y1= (
            x[:, 0],
            x[:, 1],
        )
        predicted_distances = self.forward(x).flatten()

        # Visualize the results
        plt.scatter(test_x1, test_y1, c=predicted_distances)
        plt.colorbar(label="Predicted Distance")
        plt.xlabel("X1")
        plt.ylabel("Y1")
        plt.title("Euclidean Distance between Two Points")
        plt.show()

        max_diff = 0
        print("Learning rate:", learning_rate)
        print("Batch size:", self.batch_size)
        print("Clip threshold:", self.clip_threshold)
        for i, data in enumerate(x):
            real_euclidean = np.sqrt(
                (data[2:][0] - data[:2][0]) ** 2 + (data[2:][1] - data[:2][1]) ** 2
            )
            difference = np.round(np.abs(predicted_distances[i] - real_euclidean), 2)
            if difference > max_diff:
                max_diff = difference
            print(f"{data[:2]}, {data[2:]} -> {predicted_distances[i]} (Diff: ~{difference})")
        print(f"Max difference: {max_diff}")