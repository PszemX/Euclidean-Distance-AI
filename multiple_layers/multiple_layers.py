import numpy as np
import matplotlib.pyplot as plt
from generators import Generator
from activations import *
from model import Layer, NeuralNetwork

# Generate data
num_samples = 1000
x = np.random.randint(0, 20, size=(num_samples, 2))
y = np.sum(x, axis=1)
test_data = np.random.randint(55, 100, size=(100, 2))


# Initialize model
model = NeuralNetwork()

# Add layers
model.add(Layer(2, 512, ActivationReLU()))
model.add(Layer(512, 256, ActivationReLU()))
model.add(Layer(256, 128, ActivationReLU()))
model.add(Layer(128, 1, ActivationReLU()))

# Configure and finilize model
model.configureTraining(epochs=300, batch_size=32, clip_threshold=5.0, learning_rate=0.0001)

# Train model
model.train(x, y, num_samples)

# Test model
model.test(test_data)
