import numpy as np

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig

class Perceptron():
    
    def __init__(self, number_inputs, use_bias=True):
        self.number_inputs = number_inputs
        self.use_bias = use_bias
        self.weights = [0.]*self.number_inputs
        self.bias = 0.
        

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def call(self, x): 
        y = 0
        for i, w in zip(x, self.weights):
            y = y+ i*w
        if self.use_bias:
            y = y + self.bias
        y = sigmoid(y)
        return y
    
def sigmoid_np(x):
    sig_np = 1/(1 + np.exp(-x))
    return sig_np


class MLP_layer():
    def __init__(self, num_inputs, num_units):
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.weights = np.zeros((self.num_units, self.num_inputs))
        self.bias = np.zeros((self.num_units,))

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def call(self, x):
        #x.shape: (num_inputs, 1)
        pre_activations = self.weights @ x + np.transpose(self.bias)
        activation = sigmoid_np(pre_activations) 
        return activation