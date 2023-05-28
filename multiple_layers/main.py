import numpy as np
import matplotlib.pyplot as plt
from generators import Generator
from activations import *
from optimizers import *
from model import Layer, NeuralNetwork


def configure_model(model, input):
    # Add layers
    model.add(Layer(input_size=input, output_size=512, activation=ActivationReLU()))
    model.add(Layer(input_size=512, output_size=1, activation=ActivationReLU()))
    #model.add(Layer(input_size=256, output_size=128, activation=ActivationReLU()))
    #model.add(Layer(input_size=128, output_size=1, activation=ActivationReLU()))
    #model.add(Layer(input_size=256, output_size=512, activation=ActivationReLU()))
    #model.add(Layer(input_size=512, output_size=256, activation=ActivationReLU()))
    #model.add(Layer(input_size=256, output_size=128, activation=ActivationReLU()))
    #model.add(Layer(input_size=128, output_size=1, activation=ActivationReLU()))

    # Configure and finilize model
    model.configureTraining(epochs=1000, batch_size=32, clip_threshold=5.0, accuracy=0.1, learning_rate_change_frequency=20,
                        optimizer=Optimizer_Adam(learning_rate=0.001, decay=1, beta1=0.9, beta2=0.999))

def sum_model():
    # Generate data
    generator = Generator(train_percentage='80%', dimensions=2)
    x_train, y_train, x_test, y_test = generator.generate(size=1000, min_range=-50, max_range=50, type='addition')
    # Initialize model
    model = NeuralNetwork()
    configure_model(model=model, input=2)
    # Train model
    model.train(x=x_train, y=y_train)
    # Test model
    model.test(x=x_test, y=y_test)
    return model

def sub_model():
    # Generate data
    generator = Generator(train_percentage='80%', dimensions=2)
    x_train, y_train, x_test, y_test = generator.generate(size=5000, min_range=1, max_range=100, type='substractionAB')
    print(x_train)
    print("wyniki")
    print(y_train)
    # Initialize model
    model = NeuralNetwork()
    configure_model(model=model, input=2)
    # Train model
    model.train(x=x_train, y=y_train)
    # Test model
    model.test(x=x_test, y=y_test)
    return model


def power_model():
    # Generate data
    generator = Generator(train_percentage='80%', dimensions=1)
    x_train, y_train, x_test, y_test = generator.generate(size=1000, min_range=1, max_range=100, type='power')
    # Initialize model
    model = NeuralNetwork()
    configure_model(model=model, input=1)
    # Train model
    model.train(x=x_train, y=y_train)
    # Test model
    model.test(x=x_test, y=y_test)
    return model

def sqrt_model():
    # Generate data
    generator = Generator(train_percentage='80%', dimensions=1)
    x_train, y_train, x_test, y_test = generator.generate(size=1000, min_range=1, max_range=100, type='sqrt')
    # Initialize model
    model = NeuralNetwork()
    configure_model(model=model, input=1)
    # Train model
    model.train(x=x_train, y=y_train)
    # Test model
    model.test(x=x_test, y=y_test)
    return model



sumModel = sum_model()
#subModel = sub_model()
#powerModel = power_model()
#sqrtModel = sqrt_model()

punkt1 = [1, 6]
punkt2 = [7, 9]

#odleglosc = sqrtModel.forward(sumModel.forward([powerModel.forward(subModel.forward([punkt2[0], punkt1[0]])), powerModel.forward([punkt2[1], punkt1[1]])]))