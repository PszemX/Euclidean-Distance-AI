import numpy as np
import matplotlib.pyplot as plt
from optimizers import *
from generators import *
from activations import *
from model import *

# MAIN
# Generate data
generator = Generator(train_percentage='80%', dimensions=2)
x_train, y_train, x_test, y_test = generator.generate(size=1000, min_range=-50, max_range=50, type='euklides')

# Initialize model
model = NeuralNetwork()

# Configure model
model.configure(epochs=100, batch_size=32, clip_threshold=5.0, optimizer=AdamOptimizer(), learning_rate=0.001) 
model.addLayer(Layer(input_size=4, output_size=512, activation=ActivationReLU(), optimizer=AdamOptimizer()))
model.addLayer(Layer(input_size=512, output_size=256, activation=ActivationReLU(), optimizer=AdamOptimizer()))
model.addLayer(Layer(input_size=256, output_size=128, activation=ActivationReLU(), optimizer=AdamOptimizer()))
model.addLayer(Layer(input_size=128, output_size=1, activation=ActivationReLU(), optimizer=AdamOptimizer()))

# Train model
model.train(x_train, y_train)

# Test model
model.test(x_test, y_test)

