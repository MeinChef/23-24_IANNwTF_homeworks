import numpy as np
import func

class MLP_LAYER:
    def __init__(self, in_size = 32, units = 32, activ_func = func.sigmoid):
        self.units = units
        self.in_size = in_size
        self.activ_func = activ_func

        self.bias = np.zeros(self.units)
        self.weights = np.random.normal(0, 0.2, (self.in_size, self.units))


    def set_bias(self, bias):
        assert np.shape(bias) == np.shape(self.bias), f"The function recieved a bias that isn't correct. Should have gotten an array of shape {np.shape(self.bias)}"

        self.bias = bias



    def forward(self, input):

        assert np.shape(input)[0] == self.in_size, "The function got passed a shape of inputs contradicting the setup of the layer."

        preactivation = (input @ self.weights) + self.bias
        self.activation = self.activ_func(preactivation)

        assert np.shape(self.activation)[0] == self.units, "Please check all inputs again, something went wrong."

        return self.activation

        
class MLP:
    def __init__(self, no_layer = 4, *layer_sizes):
        self.no_layer = no_layer
        assert len(layer_sizes) == no_layer, "The function got passed an amount of layer-sizes unequal to the one specified."

        self.layer_sizes = layer_sizes
        self.layer = []

        for i in range(len(self.layer_sizes)):
            if i == 0:
                self.layer.append(MLP_LAYER(in_size = 64, units = self.layer_sizes[i]))

            elif i == len(self.layer_sizes):
                self.layer.append(MLP_LAYER(in_size = self.layer_sizes[i-1], units = 10, activ_func = func.softmax))

            else:
                self.layer.append(MLP_LAYER(in_size = self.layer_sizes[i-1], units = self.layer_sizes[i]))

    def __repr__(self):
        return f"This Neuronal Network has {self.no_layer} Layers, with {self.layer_sizes} Neurons respectively."

    def cce():
        pass
        #https://gombru.github.io/2018/05/23/cross_entropy_loss/

a = MLP(2, 8, 16)
print(type(a))




