import numpy as np
import func
import mlp_layer as ml

class MLP:
    def __init__(self, num_layer = 4, layer_sizes = (8,8,8,8), learning_rate = 0.03):
        self.num_layer = num_layer

        assert len(layer_sizes) == num_layer, "The function got passed an amount of layer-sizes unequal to the one specified."
        assert type(layer_sizes) == tuple, f"The function did expect a tuple, got {type(layer_sizes)} instead"
        self.layer_sizes = layer_sizes

        self.layer = []
        self.learning_rate = learning_rate

        for i in range(self.num_layer):
            if i == 0:
                self.layer.append(ml.MLP_LAYER(in_size = 64, units = self.layer_sizes[i]))

            elif i == len(self.layer_sizes):
                self.layer.append(ml.MLP_LAYER(in_size = self.layer_sizes[i-1], units = 10, activ_func = func.softmax, func_deriv = func.cce_softmax_derivative))

            else:
                self.layer.append(ml.MLP_LAYER(in_size = self.layer_sizes[i-1], units = self.layer_sizes[i]))

    def __repr__(self):
        return f"This Neuronal Network has {self.num_layer} Layers, with {self.layer_sizes} Neurons respectively."


    def backwards(self, target):

        error0 = func.cce_softmax_derivative(self.layer[-1].activation, target)
        error = ""

        for layer in reversed(self.layer):

            if layer == self.layer[-1]:

                layer.weights_backward(error0)
                layer.update_weights(weight_gradient = layer.weight_gradient, learning_rate = self.learning_rate)

                error = layer.cross_backward(error0)
            
            else:

                error = layer.calc_error(error)

                layer.weights_backward(error)
                layer.update_weights(weight_gradient = layer.weight_gradient, learning_rate = self.learning_rate)

                error = layer.cross_backward(error)

            
    def forwards(self, input):

        for layer in self.layer:
            input = layer.forward(input)
        
        return input



        


# a = ml.MLP_LAYER(in_size = 5, units = 5)
# a.forward(np.random.normal(0, 0.2 ,(5)))
# a.weights_backward(error_signal = np.array((.2, .3, .4, .5, .5)))
# print(a)
# 


