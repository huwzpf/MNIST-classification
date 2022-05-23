import random

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNet:
    def __init__(self, layers_layout, input_size):
        self.weights = []
        self.biases = []
        self.z = []
        self.activations = []
        self.layers_count = len(layers_layout)
        self.deltas = [0 for _ in range(self.layers_count)]
        prev_layer_size = input_size
        for layer_size in layers_layout:
            self.weights.append(np.random.rand(prev_layer_size, layer_size)- 0.5)
            self.biases.append(np.zeros((1, layer_size)))
            self.z.append(np.zeros((layer_size, 1)))
            self.activations.append(np.zeros((layer_size, 1)))
            prev_layer_size = layer_size

    def cost_function(self, x, y):
        y_hat = self.predict(x)
        costs = sum(y == y_hat) / len(y_hat)
        return costs


    def forward_prop(self, x):
        self.z[0] = x.dot(self.weights[0]) + self.biases[0]
        self.activations[0] = sigmoid(self.z[0])
        for i in range(1, self.layers_count):
            self.z[i] = self.activations[i-1].dot(self.weights[i]) + self.biases[i]
            self.activations[i] = sigmoid(self.z[i])

    def backprop(self, y):
        self.deltas[-1] = self.activations[-1] - y
        for i in range(self.layers_count-2, -1, -1):
            self.deltas[i] = np.transpose(self.weights[i+1].dot(np.transpose(self.deltas[i+1])))
            self.deltas[i] = np.multiply(self.deltas[i], sigmoid_derivative(self.z[i]))

    def train_batch(self, x, y, learning_rate):
        self.forward_prop(x)
        self.backprop(y)
        self.weights[0] -= np.transpose(x).dot(self.deltas[0]) * learning_rate
        self.biases[0] -= self.deltas[0] * learning_rate
        for i in range(1, self.layers_count):
            dw = np.transpose(self.activations[i-1]).dot(self.deltas[i])
            self.weights[i] -= dw * learning_rate
            db = self.deltas[i]
            self.biases[i] -= db * learning_rate
            # print(f"\n\n{db}\n\n")

    def train(self, x, y, learning_rate=0.005, iters=100, tol=0.05):
        labels = np.zeros((len(y), 10))
        for i in range(len(y)):
            labels[i, y[i]] = 1
        prev_cost = self.cost_function(x, y)
        for c in range(iters):
            for i in range(0, y.shape[0]):
                self.train_batch(x[i, :].reshape(1, 784), labels[i, :].reshape(1, 10), learning_rate)
            cost = self.cost_function(x, y)
            print(cost)
            if abs(cost - prev_cost) < tol:
                break
            print(c)

        rnd_sample = random.sample(range(y.shape[0]), 100)
        for idx in rnd_sample:
            self.forward_prop(x[idx, :].reshape(1, 784))
            print(self.activations[-1], y[idx])

    def predict(self, x):
        self.forward_prop(x)
        return np.argmax(self.activations[-1], axis=1)
