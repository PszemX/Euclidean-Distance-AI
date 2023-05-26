import numpy as np
import matplotlib.pyplot as plt

# Generate training data
num_samples = 1000
x = np.random.randint(0, 20, size=(num_samples, 2))
y = np.sum(x, axis=1)

# Initialize neural network
input_size = 2
hidden_sizes = [256, 128, 64, 32, 16]  # List of hidden layer sizes
output_size = 1
learning_rate = 0.0001
epochs = 1000
batch_size = 32

class Activation_Relu:
    def forward(self, input):
        self.input = input
        self.output = np.maximum(input, input)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
        return self.dinputs

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros(output_size)
        self.activation = activation

    def forward(self, x):
        self.input = x
        self.output = self.activation.forward(np.dot(self.input, self.weights) + self.biases) 
        return self.output
    
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        return 0, 0

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.hidden_sizes = hidden_sizes
        self.weights = []
        self.biases = []
        # CLASS
        self.layers = []

        # from 1 to n-1 layer
        prev_size = input_size
        for size in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, size) * np.sqrt(2 / prev_size))
            self.biases.append(np.zeros(size))
            #
            self.layers.append(Layer(prev_size, size, Activation_Relu()))

            prev_size = size

        # Output layer
        self.weights.append(np.random.randn(prev_size, output_size) * np.sqrt(2 / prev_size))
        self.biases.append(np.zeros(output_size))
        # CLASS
        self.layers.append(Layer(prev_size, output_size, Activation_Relu()))

    def forward(self, x):
        self.hidden_layers = []
        prev_layer_output = x
        # CLASS
        prev_layer_output2 = x

        for i in range(len(self.layers)):
            hidden_layer = np.maximum(np.dot(prev_layer_output, self.weights[i]) + self.biases[i],
                                      np.dot(prev_layer_output, self.weights[i]) + self.biases[i])
            self.hidden_layers.append(hidden_layer)
            prev_layer_output = hidden_layer
        
        # CLASS
        for i in range(len(self.layers)):
            prev_layer_output2 = self.layers[i].forward(prev_layer_output2)

        self.output = prev_layer_output
        return self.output

    def backward(self, X, y, learning_rate):
        d_output = 2 * (self.output - y) / len(X)
        dW = []
        db = []
        dW2 = []
        db2 = []

        d_hidden = d_output
        d_hidden2 = d_output
        # zaczynamy od konca
        for i in range(len(self.hidden_sizes) - 1, -1, -1):
            # mnozymy output * pochodna outputu
            dW.append(np.dot(self.hidden_layers[i].T, d_hidden))
            db.append(np.sum(d_hidden, axis=0))
            d_hidden = np.dot(d_hidden, self.weights[i + 1].T)
            d_hidden[self.hidden_layers[i] <= 0] = 0

        dW.append(np.dot(X.T, d_hidden))
        db.append(np.sum(d_hidden, axis=0))

        # CLASS
        for i in range(len(self.hidden_sizes), -1, -1):
            dW1, dW2 = self.layers[i].backward(d_hidden2)
            dW2.append(dW1)
            db2.append(dW2)
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
