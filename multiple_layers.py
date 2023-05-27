import numpy as np
import matplotlib.pyplot as plt


class ActivationReLU:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
        return self.dinputs


class ActivationLeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        self.output = np.where(input > 0, input, self.alpha * input)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.where(self.input > 0, dvalues, self.alpha * dvalues)
        return self.dinputs


class ActivationSoftmax:
    def forward(self, input):
        self.input = input
        exp_values = np.exp(input - np.max(input, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, dvalues)
        ):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
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
                output = model.forward(batch_x)

                # Backward pass with gradient clipping
                model.backward(
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
        predicted_sums = model.forward(x).flatten()

        # Visualize the results
        plt.scatter(x[:, 0], x[:, 1], c=predicted_sums)
        plt.colorbar(label="Predicted Sum")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Sum of Two Numbers")
        plt.show()

        for i, data in enumerate(x):
            print(f"{data} -> {predicted_sums[i]}")

# Generate training data
num_samples = 1000
x = np.random.randint(0, 20, size=(num_samples, 2))
y = np.sum(x, axis=1)
# Generate test data
test_data = np.random.randint(55, 100, size=(100, 2))

# Initialize neural network
input_size = 2
hidden_sizes = [512, 256, 128, 256, 512]
output_size = 1

# Training loop
model = NeuralNetwork()

model.add(Layer(2, 512, ActivationReLU()))
model.add(Layer(512, 256, ActivationReLU()))
model.add(Layer(256, 128, ActivationReLU()))
model.add(Layer(128, 1, ActivationReLU()))

model.configureTraining(epochs=300, batch_size=32, clip_threshold=5.0, learning_rate=0.0001)
model.train(x, y, num_samples)


model.test(test_data)
