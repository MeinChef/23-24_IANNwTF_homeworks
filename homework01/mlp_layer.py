import numpy as np
import func

class MLP_LAYER:
    def __init__(self, in_size = 32, units = 32, activ_func = func.sigmoid, func_deriv = func.sigmoid_derivative):
        self.units = units
        self.in_size = in_size
        self.activ_func = activ_func
        self.func_deriv = func_deriv

        self.bias = np.zeros(self.units)
        self.weights = np.random.normal(0, 0.2, (self.in_size, self.units))


    def set_bias(self, bias):
        assert np.shape(bias) == np.shape(self.bias), f"The function recieved a bias that isn't correct. Should have gotten an array of shape {np.shape(self.bias)}"

        self.bias = bias



    def forward(self, input):
        
        assert np.shape(input)[0] == self.in_size, "The function got passed a shape of inputs contradicting the setup of the layer."

        self.input = input
        # we saved preactivation and activation in the layer itself instead of a dictionary outside - this is more intuitive to us/we think this makes more sense
        self.preactivation = (input @ self.weights) + self.bias
        self.activation = self.activ_func(self.preactivation)

        assert np.shape(self.activation)[0] == self.units, "Please check all inputs again, something went wrong."

        return self.activation

    def weights_backward(self, error_signal):

        # and we also saved the gradients in here, for consistency sake.
        self.weight_gradient = np.outer(error_signal, self.input)


    
    def cross_backward(self, error_part):

        derivativeLinput = self.weights @ error_part

        return derivativeLinput
    

    def calc_error(self, error_prev): # which should be derivativeLinput

        error = error_prev * self.func_deriv(self.preactivation)

        assert error.shape[0] == self.units

        return error

    def update_weights(self, learning_rate):
        self.weights = self.weights - learning_rate * self.weight_gradient.T

    def apply_cce(self, target):
        self.cce = func.cce(self.activation, target)
