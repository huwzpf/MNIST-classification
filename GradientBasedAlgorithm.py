from abc import ABC, abstractmethod


class GradientBasedAlgorithm(ABC):
    @abstractmethod
    def cost_function(self, x, y, reg_param):
        pass

    @abstractmethod
    def train_batch(self, x, y, learning_rate, reg_param, batch_size):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    def train(self, x, y, learning_rate=0.001, iters=100, tol=0.05, reg_param=0.001, batch_size=10):
        prev_cost = self.cost_function(x, y, reg_param)
        print(f"starting training, cost before: {prev_cost}")
        for c in range(iters):
            for i in range(0, y.shape[0]):
                if i + batch_size < y.shape[0]:
                    self.train_batch(x[i:i+batch_size, :], y[i:i+batch_size, :],
                                     learning_rate, reg_param, batch_size)
                else:
                    self.train_batch(x[i:, :].reshape(y.shape[0] - i, x.shape[1]),
                                     y[i:, :].reshape(y.shape[0] - i,  self.k),
                                     learning_rate, reg_param, y.shape[0] - i)

            cost = self.cost_function(x, y, reg_param)
            print(f"cost in {c} iteration : {cost}")
            if abs(cost - prev_cost) < tol:
                break
            prev_cost = cost