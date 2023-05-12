import tensorflow as tf
import numpy as np
from generators import Generator


#generator = Generator(train_percentage = "80%")
#a, b, c, d = generator.generate(size = 100, min_range=0, max_range=100, type="substractionAB")
#print(a)
#print(b)

#Ustawiamy ziarno losowości dla powtarzalności wyników
np.random.seed(0)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights1 = np.random.randn(self.input_size, self.hidden_size)
        self.weights2 = np.random.randn(self.hidden_size, self.output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights1))
        self.output = np.dot(self.hidden, self.weights2)
        return self.output
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def linear_derivative(self, x):
        return np.ones_like(x)
    
    def backward(self, X, y, output):
        error = y - output
        output_delta = error * self.linear_derivative(output)
        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
        self.weights1 += X.T.dot(hidden_delta) * self.learning_rate
        self.weights2 += self.hidden.T.dot(output_delta) * self.learning_rate
    
    def train(self, X, y, epochs):
        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
    
    def predict(self, X):
        return self.forward(X)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [2]])

nn = NeuralNetwork(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1)
nn.train(X, y, epochs=10000)

print(nn.predict(X))  # powinno zwrócić [[0], [1], [1], [2]]