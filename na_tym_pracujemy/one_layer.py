"""
Ten kod zawiera aktualnie najlepszą wersję dla 1 warstwy.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate training data
num_samples = 1000
x = np.random.randint(0, 20, size=(num_samples, 2))
y = np.sum(x, axis=1)

# Initialize neural network
input_size = 2
hidden_size = 512
output_size = 1
learning_rate = 0.001
epochs = 3000


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.hidden = np.maximum(
            0.01 * np.dot(X, self.W1) + self.b1, np.dot(X, self.W1) + self.b1
        )
        self.output = np.dot(self.hidden, self.W2) + self.b2
        return self.output

    def backward(self, X, y, learning_rate):
        d_output = self.output - y
        dW2 = np.dot(self.hidden.T, d_output)
        db2 = np.sum(d_output, axis=0)
        d_hidden = np.dot(d_output, self.W2.T)
        d_hidden[self.hidden <= 0] = 0
        dW1 = np.dot(X.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0)

        # Gradient clipping
        max_grad = 5.0
        dW2 = np.clip(dW2, -max_grad, max_grad)
        dW1 = np.clip(dW1, -max_grad, max_grad)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


# Training loop
model = NeuralNetwork(input_size, hidden_size, output_size)
losses = []

for epoch in range(epochs):
    # Forward pass
    output = model.forward(x)

    # Backward pass
    model.backward(x, y.reshape(-1, 1), learning_rate)

    # Print progress
    loss = np.mean((output - y.reshape(-1, 1)) ** 2)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

    # Adjust learning rate (learning rate decay)
    if (epoch + 1) % 200 == 0:
        learning_rate *= 0.1

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
