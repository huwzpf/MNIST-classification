from GradientBasedAlgorithm import GradientBasedAlgorithm
from enum import Enum
import numpy as np


class NeuralNet(GradientBasedAlgorithm):
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return NeuralNet.sigmoid(x) * (1 - NeuralNet.sigmoid(x))

    @staticmethod
    def relu(x):
        return x * (x > 0)

    @staticmethod
    def relu_derivative(x):
        return x > 0

    @staticmethod
    def logistic_loss(h, y):
        return y * np.log(h) + (1 - y) * np.log(1 - h)

    class ActivationType(Enum):
        RELU = 0,
        SIGMOID = 1

    def __init__(self, input_size, layers_layout, output_size, activation_functions=None):

        layers_layout.append(output_size)
        self.layers_count = len(layers_layout)

        self.k = output_size

        if activation_functions is None:
            self.activation_functions = [NeuralNet.ActivationType.SIGMOID for _ in range(self.layers_count)]
        else:
            if len(activation_functions) != self.layers_count:
                raise IndexError("incorrect size")
            self.activation_functions = activation_functions

        self.z = [np.empty(1) for _ in range(self.layers_count)]
        self.activations = [np.empty(1) for _ in range(self.layers_count)]
        self.biases = [np.zeros((1, layer_size)) for layer_size in layers_layout]
        self.weights = []

        self.initialize_weights(input_size, layers_layout)

    def initialize_weights(self, input_size, layers_layout):
        prev_layer_size = input_size
        for layer_size in layers_layout:
            self.weights.append(np.random.randn(prev_layer_size, layer_size) * np.sqrt(1/prev_layer_size))
            prev_layer_size = layer_size

    def activation(self, x, layer):
        if self.activation_functions[layer] == NeuralNet.ActivationType.RELU:
            return self.relu(x)
        else:
            return self.sigmoid(x)

    def activation_derivative(self, x, layer):
        if self.activation_functions[layer] == NeuralNet.ActivationType.RELU:
            return self.relu_derivative(x)
        else:
            return self.sigmoid_derivative(x)

    def cost_function(self, x, y, reg_param):
        # lambda/2 * sum of squares of all weights - l2 regularization
        rt = (reg_param / 2) * sum(np.sum(np.square(a)) for a in self.weights)
        self.forward_prop(x)
        # sum of logistic losses for all outputs
        cost = sum([NeuralNet.logistic_loss(a, b) for a, b in zip(np.nditer(self.activations[-1]), np.nditer(y))])
        return (-cost + rt)/y.shape[0]

    def forward_prop(self, x):
        self.z[0] = x.dot(self.weights[0]) + self.biases[0]
        self.activations[0] = self.activation(self.z[0], 0)
        for i in range(1, self.layers_count):
            self.z[i] = self.activations[i-1].dot(self.weights[i]) + self.biases[i]
            self.activations[i] = self.activation(self.z[i], i)

    def train_batch(self, x, y, learning_rate, reg_param, batch_size):
        self.forward_prop(x)

        # backpropagation algorithm
        deltas = [0 for _ in range(self.layers_count)]
        deltas[-1] = self.activations[-1] - y
        for i in range(self.layers_count - 2, -1, -1):
            deltas[i] = np.transpose(self.weights[i + 1].dot(np.transpose(deltas[i + 1])))
            deltas[i] = np.multiply(deltas[i], self.activation_derivative(self.z[i], i))
        #                                                    l2 regularization
        self.weights[0] -= (np.transpose(x).dot(deltas[0]) + reg_param * self.weights[0] * batch_size)\
                           * learning_rate/batch_size
        self.biases[0] -= np.sum(deltas[0], axis=0).reshape(self.biases[0].shape) * learning_rate/batch_size
        for i in range(1, self.layers_count):
            #                                                        l2 regularization
            dw = np.transpose(self.activations[i-1]).dot(deltas[i]) + reg_param * self.weights[i] * batch_size
            self.weights[i] -= dw * learning_rate/batch_size
            db = np.sum(deltas[i], axis=0).reshape(self.biases[i].shape)
            self.biases[i] -= db * learning_rate/batch_size

    def predict(self, x):
        self.forward_prop(x)
        return np.argmax(self.activations[-1], axis=1)
