import numpy as np
import matplotlib.pyplot as plt
from GradientBasedAlgorithm import GradientBasedAlgorithm


class SoftmaxRegression(GradientBasedAlgorithm):
    def __init__(self, input_size, output_size):
        self.Theta = np.zeros((input_size + 1, output_size))
        self.k = output_size

    def cost_function(self, x, y, reg_param):
        rt = np.sqrt(sum(sum(reg_param * np.power(self.Theta, 2))))
        # (m as a constant in order to avoid rounding errors, without it cost ~= e-5)
        ret = (np.sum((np.sum(np.multiply(y, self.hypothesis(np.c_[np.ones(x.shape[0]), x])), axis=1))) + rt) / y.shape[0]
        return 1 / ret

    def hypothesis(self, x):
        h = np.exp(x.dot(self.Theta))
        c = (1 / np.sum(h, axis=1)).reshape(x.shape[0], 1)
        return np.multiply(c, h)

    def predict(self, x):
        return np.argmax(self.hypothesis(np.c_[np.ones(x.shape[0]), x]))

    def gradient(self, x, y, reg_term):
        rt = (reg_term * self.Theta)
        rt[0, :] = 0
        return np.transpose(x).dot(self.hypothesis(x) - y) + rt

    def train_batch(self, x, y, learning_rate, reg_param, batch_size):
        self.Theta -= learning_rate * self.gradient(np.c_[np.ones(x.shape[0]), x], y, reg_param) / y.shape[0]

    def print_theta_as_square_matrix(self, t_size=28):
        for i in range(self.Theta.shape[1]):
            w = np.transpose(self.Theta[1:, i]).reshape(t_size, t_size)
            plt.imshow(w, cmap='hot', interpolation='nearest')
            plt.title(f"Theta for {i} class")
            plt.show()










