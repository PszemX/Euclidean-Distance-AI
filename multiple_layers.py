"""
Ten kod zawiera aktualnie najlepszą wersję dla wielu warstw.
"""


import numpy as np
import matplotlib.pyplot as plt

# Generate training data
num_samples = 1000
x = np.random.randint(0, 20, size=(num_samples, 2))
y = np.sum(x, axis=1)

# Initialize neural network
input_size = 2
hidden_sizes = [512, 256, 128]  # List of hidden layer sizes
output_size = 1
learning_rate = 0.0001
epochs = 3000
batch_size = 32


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.hidden_sizes = hidden_sizes
        self.weights = []
        self.biases = []

        # from 1 to n-1 layer
        prev_size = input_size
        for size in hidden_sizes:
            self.weights.append(
                np.random.randn(prev_size, size) * np.sqrt(2 / prev_size)
            )
            self.biases.append(np.zeros(size))
            prev_size = size

        # Output layer
        self.weights.append(
            np.random.randn(prev_size, output_size) * np.sqrt(2 / prev_size)
        )
        self.biases.append(np.zeros(output_size))

    def forward(self, X):
        self.hidden_layers = []
        prev_layer_output = X

        for i in range(len(self.hidden_sizes)):
            hidden_layer = np.maximum(
                0.01 * np.dot(prev_layer_output, self.weights[i]) + self.biases[i],
                np.dot(prev_layer_output, self.weights[i]) + self.biases[i],
            )
            self.hidden_layers.append(hidden_layer)
            prev_layer_output = hidden_layer

        self.output = np.dot(prev_layer_output, self.weights[-1]) + self.biases[-1]
        return self.output

    def backward(self, X, y, learning_rate):
        d_output = 2 * (self.output - y) / len(X)
        dW = []
        db = []

        d_hidden = d_output

        for i in range(len(self.hidden_sizes) - 1, -1, -1):
            dW.append(np.dot(self.hidden_layers[i].T, d_hidden))
            db.append(np.sum(d_hidden, axis=0))
            d_hidden = np.dot(d_hidden, self.weights[i + 1].T)
            d_hidden[self.hidden_layers[i] <= 0] = 0

        dW.append(np.dot(X.T, d_hidden))
        db.append(np.sum(d_hidden, axis=0))

        # Gradient clipping
        max_grad = 1.0
        dW = [np.clip(dw, -max_grad, max_grad) for dw in dW]
        db = [np.clip(db, -max_grad, max_grad) for db in db]

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[len(dW) - 1 - i]
            self.biases[i] -= learning_rate * db[len(db) - 1 - i]


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

        # Backward pass
        model.backward(batch_x, batch_y.reshape(-1, 1), learning_rate)

        # Compute batch loss
        batch_loss = np.mean((output - batch_y.reshape(-1, 1)) ** 2)
        epoch_loss += batch_loss

    # Print progress
    loss = epoch_loss / (num_samples // batch_size)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

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
