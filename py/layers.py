import numpy as np

def rand_matrix(m, n):
    array = np.random.rand(m, n) - 1/np.float(2)
    return np.matrix(array)

class Layer():
    def __init__(self, units, input=None, activation=None):
        self.units = units
        self.activation = activation

    def __call__(self, inputs, training=False):
        outputs = inputs

        if training != False and self.activation != None:
            outputs = self.activation.function(outputs)

        return outputs

    def getUnits(self):
        return self.units

    def get_function(self):
        return self.activation


class InputLayer(Layer):
    def __init__(self, units):
        super().__init__(units)

    def __call__(self, inputs):
        return np.array(inputs)

class Dense(Layer):
    def __init__(self, units, input, activation=None):
        super().__init__(units, input, activation)
        self.weights = rand_matrix(units, input.getUnits())
        self.biases = rand_matrix(units, 1)

    def __call__(self, inputs, training=False):
        outputs = np.matrix(inputs)
        outputs = np.add(self.weights.dot(outputs.T), self.biases)

        return super().__call__(outputs.T, training)

    def apply(self, gradient, weight_delta):
        self.biases += gradient.T
        self.weights += weight_delta

    def get_weights(self):
        return self.weights

