import numpy as np
import math
# z ksiazki
class NeuralNetwork():

    def __init__(self, n_input, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # ReLU activation function for values < 0 return 0, otherwise return value
    def activation_ReLU(self, X):
        return np.maximum(0, X)
    
    # Linear activation function for values return values, it is also named non activation function
    def adctivation_linear(self, X):
        return np.ones_like(X)
        
    # Sigmoid activation function return count from pattern, return between 0 and 1
    def adctivation_sigmoid(self, X):
        return X/(1 - ((math.e)**X))
    
    # Softmax function is normalize exponential function, with this we can predict output
    # for each output value normalizes to a fraction of the sum, all of the values are now in the range of 0 to 1 and add up to 1
    # exp(): y = e**x
    # return exp(x)/(np.sum(exp(X))) - easy overflow, beacue exponential function grows very fast
    # to fix it we using np.exp(X - np.max(...))
    def activation_softmax(self, X):
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    # Coutnig loss of model
    # function take prediction and amount of inputs of the neuron and calculate loss
    # include resolve to errors
    def loss(self, output, y_true):
        # Number of samples in a batch
        samples = len(output)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(output, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        sample_losses = negative_log_likelihoods
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss
    
    # Define accurancy of model 
    def accurancy(self, output, y_true):
        pass

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        # Simply output input * weight + bias
        self.output = self.activation_sigmoid(np.dot(inputs, self.weights) + self.biases)
