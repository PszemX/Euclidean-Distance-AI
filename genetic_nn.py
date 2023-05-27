import threading
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
epochs = 1000
batch_size = 32
clip_threshold = 5.0  # Adjust the threshold as needed


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


# Genetic Algorithm Parameters
population_size = 20
elite_percentage = 0.2
generations = 100
mutation_rate = 0.1


def create_individual():
    initial_learning_rate = 10 ** np.random.randint(-5, -1)
    individual = {
        "hidden_sizes": [
            np.random.choice([16, 32, 64, 128, 256, 512, 1024])
            for _ in range(np.random.randint(1, 6))
        ],
        "initial_learning_rate": initial_learning_rate,
        "learning_rate": initial_learning_rate,
        "batch_size": np.random.choice([16, 32, 64, 128]),
        "clip_threshold": np.random.uniform(0.5, 10.0),
        "fitness": None,
    }
    return individual


def evaluate_individual(individual):
    # Reinitialize the neural network with the current hyperparameters
    model = NeuralNetwork(input_size, individual["hidden_sizes"], output_size)

    # Train the model using the current hyperparameters
    for epoch in range(epochs):
        epoch_loss = 0.0

        indices = np.random.permutation(num_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        for idx in range(0, num_samples, individual["batch_size"]):
            batch_x = x_shuffled[idx : idx + individual["batch_size"]]
            batch_y = y_shuffled[idx : idx + individual["batch_size"]]

            output = model.forward(batch_x)
            model.backward(
                batch_x,
                batch_y.reshape(-1, 1),
                individual["learning_rate"],
                clip_threshold=individual["clip_threshold"],
            )

            batch_loss = np.mean((output - batch_y.reshape(-1, 1)) ** 2)
            epoch_loss += batch_loss

        loss = epoch_loss / (num_samples // individual["batch_size"])
        if (epoch + 1) % 200 == 0:
            individual["learning_rate"] *= 0.1
        if np.isnan(loss):
            fitness = np.inf  # Set fitness to negative infinity for NaN loss
            break
        fitness = loss  # Negative loss value as fitness
    print(f"{individual} -> Final loss: {loss}")
    return fitness


def evaluate_individual_thread(individual):
    individual["fitness"] = evaluate_individual(individual)


def select_parents(population):
    fitnesses = np.array([ind["fitness"] for ind in population])
    probabilities = fitnesses / np.sum(fitnesses)
    parent_indices = np.random.choice(
        len(population), size=2, replace=False, p=probabilities
    )
    parents = [population[idx] for idx in parent_indices]
    return parents


def crossover(parent1, parent2):
    child = {
        "hidden_sizes": parent1["hidden_sizes"].copy(),
        "initial_learning_rate": parent1["initial_learning_rate"],
        "learning_rate": parent1["initial_learning_rate"],
        "batch_size": parent1["batch_size"],
        "clip_threshold": parent1["clip_threshold"],
        "fitness": None,
    }

    child_hidden_sizes = []
    # Determine the number of hidden layers for the child
    num_layers = min(len(parent1["hidden_sizes"]), len(parent2["hidden_sizes"]))

    # Randomly select hidden layer sizes from both parents
    for i in range(num_layers):
        if np.random.rand() < 0.5:
            child_hidden_sizes.append(parent1["hidden_sizes"][i])
        else:
            child_hidden_sizes.append(parent2["hidden_sizes"][i])

    # Add any remaining hidden layer sizes from the longer parent
    if len(parent1["hidden_sizes"]) > num_layers:
        child_hidden_sizes.extend(parent1["hidden_sizes"][num_layers:])
    elif len(parent2["hidden_sizes"]) > num_layers:
        child_hidden_sizes.extend(parent2["hidden_sizes"][num_layers:])

    # Create the child individual with the new hidden sizes
    child["hidden_sizes"] = child_hidden_sizes
    return child


def mutate(individual):
    if np.random.uniform() < 0.5:
        # Mutate hidden sizes
        for i in range(len(individual["hidden_sizes"])):
            if np.random.uniform() < mutation_rate:
                individual["hidden_sizes"][i] = np.random.choice(
                    [16, 32, 64, 128, 256, 512, 1024]
                )
    else:
        # Mutate learning rate
        if np.random.uniform() < mutation_rate:
            mutated_learning_rate = 10 ** np.random.randint(-5, -1)
            individual["learning_rate"] = mutated_learning_rate
            individual["initial_learning_rate"] = mutated_learning_rate

        # Mutate batch size
        if np.random.uniform() < mutation_rate:
            individual["batch_size"] = np.random.choice([16, 32, 64, 128])

        # Mutate clip threshold
        if np.random.uniform() < mutation_rate:
            individual["clip_threshold"] = np.random.uniform(0.5, 10.0)

    return individual


# Initialize population
population = [create_individual() for _ in range(population_size)]
print("Create individuals")

# Evolutionary loop
for generation in range(generations):
    # Evaluate fitness using threads for parallel evaluation
    threads = []
    for individual in population:
        thread = threading.Thread(target=evaluate_individual_thread, args=(individual,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Sort population based on fitness
    population = sorted(population, key=lambda x: x["fitness"])

    # Print best individual in the current generation
    best_individual = population[0]
    print(f"Generation: {generation}, Best Fitness: {best_individual['fitness']:.8f}")
    print("Best Individual - Hyperparameters:")
    print(f"Hidden Sizes: {best_individual['hidden_sizes']}")
    print(f"Initial Learning Rate: {best_individual['initial_learning_rate']}")
    print(f"Batch Size: {best_individual['batch_size']}")
    print(f"Clip Threshold: {best_individual['clip_threshold']}")
    print("----------------------------------------")

    # Select elite individuals
    elite_count = int(population_size * elite_percentage)
    elite = population[:elite_count]

    # Perform crossover and mutation
    offspring = []
    for _ in range(population_size - elite_count):
        parent1, parent2 = select_parents(population)
        child = crossover(parent1, parent2)
        child = mutate(child)
        offspring.append(child)

    # Create new population
    population = elite + offspring

# Evaluate fitness of the final population
for individual in population:
    individual["fitness"] = evaluate_individual(individual)

# Sort population based on fitness
population = sorted(population, key=lambda x: x["fitness"])

# Print best individual in the final population
best_individual = population[0]
print("Final Results:")
print(f"Best Individual - Fitness: {best_individual['fitness']:.8f}")
print("Best Individual - Hyperparameters:")
print(f"Hidden Sizes: {best_individual['hidden_sizes']}")
print(f"Initial Learning Rate: {best_individual['initial_learning_rate']}")
print(f"Batch Size: {best_individual['batch_size']}")
print(f"Clip Threshold: {best_individual['clip_threshold']}")
print("----------------------------------------")

# Reinitialize the neural network with the best hyperparameters
model = NeuralNetwork(input_size, best_individual["hidden_sizes"], output_size)

# Train the model using the best hyperparameters
for epoch in range(epochs):
    epoch_loss = 0.0

    indices = np.random.permutation(num_samples)
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    for idx in range(0, num_samples, best_individual["batch_size"]):
        batch_x = x_shuffled[idx : idx + best_individual["batch_size"]]
        batch_y = y_shuffled[idx : idx + best_individual["batch_size"]]

        output = model.forward(batch_x)
        model.backward(
            batch_x,
            batch_y.reshape(-1, 1),
            best_individual["initial_learning_rate"],
            clip_threshold=best_individual["clip_threshold"],
        )

        batch_loss = np.mean((output - batch_y.reshape(-1, 1)) ** 2)
        epoch_loss += batch_loss

    loss = epoch_loss / (num_samples // best_individual["batch_size"])
    print(f"Epoch: {epoch+1}, Loss: {loss:.8f}")

# Plot the training data and the predictions
x_test = np.random.randint(0, 20, size=(100, 2))
y_test = np.sum(x_test, axis=1)
predictions = model.forward(x_test)
predictions = predictions.reshape(-1)

plt.scatter(x_test[:, 0], x_test[:, 1], c="b", label="Input")
plt.scatter(x_test[:, 0], x_test[:, 1], c="r", label="Prediction")
plt.scatter(x_test[:, 0], x_test[:, 1], c="g", label="Ground Truth")
plt.legend()
plt.show()
