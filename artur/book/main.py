from generators import Generator
from Model import *
from Optimizer import *
from Activations import *
import numpy as np

MAX_VALUE_INPUT = 100
MAX_VALUE_OUTPUT = MAX_VALUE_INPUT**2

# Create dataset
generator = Generator('80%', 2)
x_train, y_train, x_test, y_test = generator.generate(6000, 1, MAX_VALUE_INPUT, 'multiplication')

x_train = (x_train.reshape(x_train.shape[0], -1).astype(np.float32) - (MAX_VALUE_INPUT/2)) / (MAX_VALUE_INPUT/2)
x_test = (x_test.reshape(x_test.shape[0], -1).astype(np.float32) - (MAX_VALUE_INPUT/2)) / (MAX_VALUE_INPUT/2)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

y_train = y_train.reshape(y_train.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

#y_train = (y_train.reshape(y_train.shape[0], -1).astype(np.float32) - (MAX_VALUE_OUTPUT/2)) / (MAX_VALUE_OUTPUT/2)
#y_test = (y_test.reshape(y_test.shape[0], -1).astype(np.float32) - (MAX_VALUE_OUTPUT/2)) / (MAX_VALUE_OUTPUT/2)
# Create model
model = Model()
# Add layers
model.add(Layer_Dense(x_train.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 1))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(x_train, y_train, validation_data=(x_test, y_test),
            epochs=10, print_every=100)

# Evaluate the model
model.evaluate(x_test, y_test)
