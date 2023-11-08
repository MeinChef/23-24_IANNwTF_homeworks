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
        self.preactivation = (input @ self.weights) + self.bias
        self.activation = self.activ_func(self.preactivation)

        assert np.shape(self.activation)[0] == self.units, "Please check all inputs again, something went wrong."

        #return self.activation

    def weights_backward(self, error_signal):

        derivativeLW = np.outer(error_signal, self.input)
        derivativeLinput = error_signal * self.weights

        return derivativeLW, derivativeLinput

    def calc_error(self, error_prev): # which should be derivativeLinput
        
        error = error_prev * self.func_deriv(self.input)

        assert [1] == self.units

        return error


    def update_weights(self, weight_gradient, learning_rate = 0.03):
        self.weights = self.weights - learning_rate * weight_gradient


    # update weights = old weights - learning rate * green arrow
        
class MLP:
    def __init__(self, num_layer = 4, *layer_sizes):
        self.num_layer = num_layer
        assert len(layer_sizes) == num_layer, "The function got passed an amount of layer-sizes unequal to the one specified."

        self.layer_sizes = layer_sizes
        self.layer = []

        for i in range(self.num_layer):
            if i == 0:
                self.layer.append(MLP_LAYER(in_size = 64, units = self.layer_sizes[i]))

            elif i == len(self.layer_sizes):
                self.layer.append(MLP_LAYER(in_size = self.layer_sizes[i-1], units = 10, activ_func = func.softmax, func_deriv = func.cce_softmax_derivative))

            else:
                self.layer.append(MLP_LAYER(in_size = self.layer_sizes[i-1], units = self.layer_sizes[i]))

    def __repr__(self):
        return f"This Neuronal Network has {self.num_layer} Layers, with {self.layer_sizes} Neurons respectively."


    def backwards(self, target):

        error = func.cce_softmax_derivative(self.layer[-1].activation, target)

        for layer in reversed(self.layer):

            if layer == self.layer[-1]:
                weight_gradient, error = layer.weights_backwards(error)
                layer.update_weights(weight_gradient = weight_gradient)
            
            else:
                error = layer.calc_error(error)
                weight_gradient, error = layer.weights_backwards(error)
                layer.update_weights(weight_gradient = weight_gradient)



        


a = MLP_LAYER(in_size = 5, units = 5)
a.forward(np.random.normal(0, 0.2 ,(5)))
a.weights_backward(error_signal = np.array((.2, .3, .4, .5, .5)))
print(a)



