import numpy as np
from .activation import sigmoid

class Trainer():
    def __init__(self, lr=0.01):
        self.learning_rate = lr
        self.outs, self.inps, self.layers = [], [], []

    def __call__(self, layer):
        self.compute(layer, self.outs[len(self.outs) - 1], compute=True)

    def compute(self, layer, inputs, compute=False):
        outputs, net = None, None

        if compute == False:
            outputs = layer(inputs)
            net = []
        else:
            none_function = layer(inputs, training=True)
            function = layer.get_function()
            outputs = function(none_function)
            net = function.derivative(none_function)

        self.outs.append(outputs)
        self.inps.append(net)
        self.layers.append(layer)

    def reset(self):
        self.outs, self.inps, self.layers = [], [], []

    def get_outputs(self):
        return self.outs[len(self.outs) - 1]

    #backprop is unfinished
    def backprop(self, targets):
        # outputs = self.get_outputs()
        # outputs_error = np.array(targets) - outputs
        # layer_i = len(self.outs) - 1

        # gradient = np.multiply(self.inps[layer_i], outputs_error) * self.learning_rate # gradient = a'(net_j) * (o_j - t_j)
        # self.layers[layer_i].apply(gradient, gradient * np.matrix(self.outs[layer_i]).T)

        # sum_dw = outputs_error * self.layers[layer_i].get_weights() # sum k = d_k * w_jk

        # hidden_length = layer_i - 1
        # if hidden_length > 0:
        #     for j in range(hidden_length):
        #         i = hidden_length - j

        #         if i == 0:
        #             break

        #         gradient = np.multiply(self.inps[i], sum_dw) * self.learning_rate
        #         self.layers[i].apply(gradient, gradient * np.matrix(self.outs[i]).T)

        #         if i - 1 > 0:
        #             sum_dw = sum_dw * self.layers[i].get_weights()
