import numpy as np
from generators import Generator
from NeuralNetwork import NeuralNetwork


# Definiujemy funkcję aktywacji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Ustawiamy ziarno losowości dla powtarzalności wyników CHAT GPT DZIALA I DOSYC DOBRE JEST
np.random.seed(0)


# Definiujemy klasę NeuralNetwork, która będzie reprezentować naszą sieć neuronową
class NeuralNetwork2:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicjujemy wagi losowo
        self.weights1 = np.random.rand(self.input_size, self.hidden_size)
        self.weights2 = np.random.rand(self.hidden_size, self.output_size)

    def forward(self, X):
        # Propagujemy sygnał przez warstwę ukrytą
        self.hidden = sigmoid(np.dot(X, self.weights1))

        # Propagujemy sygnał przez warstwę wyjściową
        output = sigmoid(np.dot(self.hidden, self.weights2))
        return output

    def backward(self, X, y, output, learning_rate):
        # Liczymy błąd
        error = y - output

        # Obliczamy gradienty dla warstwy wyjściowej
        output_grad = error * (output * (1 - output))

        # Obliczamy gradienty dla warstwy ukrytej
        hidden_grad = self.hidden * (1 - self.hidden)
        hidden_grad = np.dot(output_grad, self.weights2.T) * hidden_grad

        # Aktualizujemy wagi
        self.weights2 += learning_rate * np.dot(self.hidden.T, output_grad)
        self.weights1 += learning_rate * np.dot(X.T, hidden_grad)

    def train(self, X, y, learning_rate, epochs):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)

    def predict(self, X):
        return self.forward(X)


# Tworzymy dane treningowe
X = np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])
y = np.array([[0], [0], [0], [0], [1], [1], [2], [3]])

# Tworzymy sieć neuronową z jedną warstwą ukrytą o 3 neuronach
nn = NeuralNetwork(2, 3, 1)

# Uczymy sieć neuronową na danych treningowych
nn.train(X, y, 0.1, 10000)

# Testujemy nauczoną sieć neuronową
print(nn.predict(np.array([4, 5])))  # powinno zwrócić około 1

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
