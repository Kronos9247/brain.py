import numpy as np

class ActivationFunction():
    def __init__(self, func, dfunc):
        self.function = func
        self.derivative = dfunc

    def __call__(self, input):
        return self.function(input)

    def function(input):
        return self.function(input)

    def derivative(input):
        return self.derivative(input)

def sigmoid_func(t):
    return 1 / (1 + np.exp(-t))

def sigmoid_deriv(t):
    output = sigmoid_func(t)
    return np.multiply(output, (1 - output))

sigmoid = ActivationFunction(sigmoid_func, sigmoid_deriv)