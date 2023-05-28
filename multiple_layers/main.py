import numpy as np
import matplotlib.pyplot as plt
from generators import Generator
from activations import *
from optimizers import *
from model import Layer, NeuralNetwork

# Generate data
generator = Generator(train_percentage='80%', dimensions=2)
x_train, y_train, x_test, y_test = generator.generate(size=1000, min_range=1, max_range=100, type='addition')

# Initialize model
model = NeuralNetwork()

# Add layers
model.add(Layer(input_size=2, output_size=512, activation=ActivationReLU()))
model.add(Layer(input_size=512, output_size=256, activation=ActivationReLU()))
model.add(Layer(input_size=256, output_size=128, activation=ActivationReLU()))
model.add(Layer(input_size=128, output_size=1, activation=ActivationReLU()))

# Configure and finilize model
model.configureTraining(epochs=600, batch_size=64, clip_threshold=5.0, accuracy=0.1, learning_rate_change_frequency=10,
                        optimizer=Optimizer_Adam(learning_rate=0.001, decay=1, beta1=0.9, beta2=0.999))

# Train model
model.train(x=x_train, y=y_train)

# Test model
model.test(x=x_test, y=y_test)
