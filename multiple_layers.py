import numpy as np
import matplotlib.pyplot as plt

# Generate training data
num_samples = 1000
x = np.random.randint(0, 20, size=(num_samples, 2))
y = np.sum(x, axis=1)

# Initialize neural network
input_size = 2
hidden_sizes = [512, 256, 128]
output_size = 1
learning_rate = 0.0001
epochs = 3000
batch_size = 32
clip_threshold = 1.0  # Adjust the threshold as needed


class ActivationReLU:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
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
    def __init__(self, input_size, hidden_sizes, output_size):
        self.hidden_sizes = hidden_sizes
        self.layers = []

        # Input layer to first hidden layer
        self.layers.append(Layer(input_size, hidden_sizes[0], ActivationReLU()))

        # Hidden layers with batch normalization
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                Layer(hidden_sizes[i - 1], hidden_sizes[i], ActivationReLU())
            )

        # Last hidden layer to output layer
        self.layers.append(Layer(hidden_sizes[-1], output_size, ActivationReLU()))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y, learning_rate, clip_threshold=None):
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
                layer.weights -= learning_rate * layer.dweights
                layer.biases -= learning_rate * layer.dbiases


# Training loop
model = NeuralNetwork(input_size, hidden_sizes, output_size)
losses = []

for epoch in range(epochs):
    epoch_loss = 0.0

    # Shuffle training data
    indices = np.random.permutation(num_samples)
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    for idx in range(0, num_samples, batch_size):
        batch_x = x_shuffled[idx : idx + batch_size]
        batch_y = y_shuffled[idx : idx + batch_size]

        # Forward pass
        output = model.forward(batch_x)

        # Backward pass with gradient clipping
        model.backward(
            batch_x,
            batch_y.reshape(-1, 1),
            learning_rate,
            clip_threshold=clip_threshold,
        )

        # Compute batch loss
        batch_loss = np.mean((output - batch_y.reshape(-1, 1)) ** 2)
        epoch_loss += batch_loss

    # Print progress
    loss = epoch_loss / (num_samples // batch_size)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.8f}")

    # Adjust learning rate (learning rate decay)
    if (epoch + 1) % 200 == 0:
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

# Generate test data
test_data = np.random.randint(0, 20, size=(100, 2))
predicted_sums = model.forward(test_data).flatten()

# Visualize the results
plt.scatter(test_data[:, 0], test_data[:, 1], c=predicted_sums)
plt.colorbar(label="Predicted Sum")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Sum of Two Numbers")
plt.show()

for i, data in enumerate(test_data):
    print(f"{data} -> {predicted_sums[i]}")
