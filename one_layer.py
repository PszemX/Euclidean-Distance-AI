import numpy as np
import matplotlib.pyplot as plt

# Generate training data
num_samples = 10000
x = np.random.randint(-100, 100, size=(num_samples, 4))  # [x1, y2, x2, y2]
y = np.array(
    [np.linalg.norm(np.array((xy[0], xy[1])) - np.array((xy[2], xy[3]))) for xy in x]
)

# Initialize neural network
input_size = 4
hidden_size = 1024
output_size = 1
learning_rate = 0.01
epochs = 3000
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(
            2 / input_size
        )
        self.biases = np.zeros(output_size)
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)

    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output

    def backward(self, d_output, learning_rate):
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0)

        # Gradient clipping
        max_grad = 1.0
        d_weights = np.clip(d_weights, -max_grad, max_grad)

        # Adam optimization
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * d_weights
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (d_weights**2)
        m_weights_hat = self.m_weights / (1 - beta1)
        v_weights_hat = self.v_weights / (1 - beta2)
        self.weights -= (
            learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)
        )

        self.m_biases = beta1 * self.m_biases + (1 - beta1) * d_biases
        self.v_biases = beta2 * self.v_biases + (1 - beta2) * (d_biases**2)
        m_biases_hat = self.m_biases / (1 - beta1)
        v_biases_hat = self.v_biases / (1 - beta2)
        self.biases -= learning_rate * m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.layer1 = Layer(input_size, hidden_size)
        self.layer2 = Layer(hidden_size, output_size)

    def forward(self, X):
        hidden_output = self.layer1.forward(X)
        self.output = self.layer2.forward(hidden_output)
        return self.output

    def backward(self, X, y, learning_rate):
        d_output = self.output - y
        d_hidden = np.dot(d_output, self.layer2.weights.T)
        d_hidden[self.layer1.output <= 0] = 0

        self.layer2.backward(d_output, learning_rate)
        self.layer1.backward(d_hidden, learning_rate)


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
        print(f"Epoch: {epoch}, Loss: {loss:.8f}")

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
