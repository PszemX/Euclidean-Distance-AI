import numpy as np
import matplotlib.pyplot as plt

# Generate training data
num_samples = 1000
x = np.random.randint(-100, 100, size=(num_samples, 4))
x1, y1, x2, y2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
y = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Initialize neural network
input_size = 4  # Two points (x1, y1) and (x2, y2)
hidden_sizes = [512, 256, 128]
output_size = 1
learning_rate = 0.001
epochs = 1000
batch_size = 32
clip_threshold = 5.0  # Adjust the threshold as needed


# Adam optimizer
class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

    def update(self, weights, gradients, m, v):
        self.t += 1

        m = self.beta1 * m + (1 - self.beta1) * gradients
        v = self.beta2 * v + (1 - self.beta2) * (gradients**2)

        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)

        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights, m, v


class ActivationLeakyReLU:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(input, 0.01 * input)  # Leaky ReLU activation
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


class Layer:
    def __init__(self, input_size, output_size, activation, optimizer=None):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2 / input_size
        )
        self.biases = np.zeros(output_size)
        self.activation = activation
        self.dweights = None
        self.dbiases = None
        self.optimizer = optimizer
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.activation.forward(self.output)

    def backward(self, dvalues):
        dinputs = self.activation.backward(dvalues)
        self.dweights = np.dot(self.input.T, dinputs)
        self.dbiases = np.sum(dinputs, axis=0)

        if self.optimizer:
            self.weights, self.m_weights, self.v_weights = self.optimizer.update(
                self.weights, self.dweights, self.m_weights, self.v_weights
            )
            self.biases, self.m_biases, self.v_biases = self.optimizer.update(
                self.biases, self.dbiases, self.m_biases, self.v_biases
            )
        else:
            self.weights -= learning_rate * self.dweights
            self.biases -= learning_rate * self.dbiases

        return np.dot(dinputs, self.weights.T)


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.hidden_sizes = hidden_sizes
        self.layers = []

        # Input layer to first hidden layer
        self.layers.append(
            Layer(input_size, hidden_sizes[0], ActivationLeakyReLU(), AdamOptimizer())
        )

        # Hidden layers with batch normalization
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                Layer(
                    hidden_sizes[i - 1],
                    hidden_sizes[i],
                    ActivationLeakyReLU(),
                    AdamOptimizer(),
                )
            )

        # Last hidden layer to output layer
        self.layers.append(
            Layer(hidden_sizes[-1], output_size, ActivationLeakyReLU(), AdamOptimizer())
        )

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
test_data = np.random.randint(-100, 100, size=(100, 4))
test_x1, test_y1, test_x2, test_y2 = (
    test_data[:, 0],
    test_data[:, 1],
    test_data[:, 2],
    test_data[:, 3],
)
predicted_distances = model.forward(test_data).flatten()

# Visualize the results
plt.scatter(test_x1, test_y1, c=predicted_distances)
plt.colorbar(label="Predicted Distance")
plt.xlabel("X1")
plt.ylabel("Y1")
plt.title("Euclidean Distance between Two Points")
plt.show()

max_diff = 0
for i, data in enumerate(test_data):
    real_euclidean = np.sqrt(
        (data[2:][0] - data[:2][0]) ** 2 + (data[2:][1] - data[:2][1]) ** 2
    )
    difference = np.round(np.abs(predicted_distances[i] - real_euclidean), 2)
    if difference > max_diff:
        max_diff = difference
    print(f"{data[:2]}, {data[2:]} -> {predicted_distances[i]} (Diff: ~{difference})")
print(f"Max difference: {max_diff}")
