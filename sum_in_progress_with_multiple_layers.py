import numpy as np
import matplotlib.pyplot as plt

# Generate training data
num_samples = 1000
x = np.random.randint(0, 20, size=(num_samples, 2))
y = np.sum(x, axis=1)

# Initialize neural network
input_size = 2
hidden_size = [8, 4, 2]  # List of hidden layer sizes
output_size = 1
learning_rate = 0.001
epochs = 3000

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layers = len(hidden_size)
        self.W = []
        self.b = []
        self.output_size = output_size

        self.W.append(np.random.randn(input_size, hidden_size[0]) * np.sqrt(2 / input_size))
        self.b.append(np.zeros(hidden_size[0]))

        for i in range(1, self.hidden_layers):
            self.W.append(np.random.randn(hidden_size[i-1], hidden_size[i]) * np.sqrt(2 / hidden_size[i-1]))
            self.b.append(np.zeros(hidden_size[i]))

        self.W.append(np.random.randn(hidden_size[-1], output_size) * np.sqrt(2 / hidden_size[-1]))
        self.b.append(np.zeros(output_size))

    def forward(self, X):
        self.hidden = [0] * (self.hidden_layers + 1)
        self.hidden[0] = np.maximum(0.01 * np.dot(X, self.W[0]) + self.b[0], np.dot(X, self.W[0]) + self.b[0])

        for i in range(1, self.hidden_layers):
            self.hidden[i] = np.maximum(0.01 * np.dot(self.hidden[i-1], self.W[i]) + self.b[i],
                                        np.dot(self.hidden[i-1], self.W[i]) + self.b[i])

        self.output = np.dot(self.hidden[-1], self.W[-1]) + self.b[-1]
        return self.output

    def backward(self, X, y, learning_rate):
        d_output = self.output - y

        dW = [0] * (self.hidden_layers + 1)
        db = [0] * (self.hidden_layers + 1)

        dW[-1] = np.dot(self.hidden[-1].T, d_output)
        db[-1] = np.sum(d_output, axis=0)

        d_hidden = np.dot(d_output, self.W[-1].T)
        d_hidden[self.hidden[-1] <= 0] = 0

        for i in range(self.hidden_layers - 2, -1, -1):
            dW[i] = np.dot(self.hidden[i].T, d_hidden)
            db[i] = np.sum(d_hidden, axis=0)

            d_hidden = np.dot(d_hidden, self.W[i+1].T)
            d_hidden[self.hidden[i] <= 0] = 0

        dW[0] = np.dot(X.T, d_hidden)
        db[0] = np.sum(d_hidden, axis=0)

        # Gradient clipping
        max_grad = 5.0
        for i in range(self.hidden_layers + 1):
            dW[i] = np.clip(dW[i], -max_grad, max_grad)

        # Update weights and biases
        for i in range(self.hidden_layers + 1):
            self.W[i] -= learning_rate * dW[i]
            self.b[i] -= learning_rate * db[i]

# Training loop
model = NeuralNetwork(input_size, hidden_size, output_size)
losses = []

for epoch in range(epochs):
    # Forward pass
    output = model.forward(x)

    # Reshape the output and target to have the same shape
    output = output.reshape(-1, output_size)
    target = y.reshape(-1, output_size)

    # Backward pass
    model.backward(x, target, learning_rate)

    # Print progress
    loss = np.mean((output - target) ** 2)
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
