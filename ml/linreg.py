import numpy as np
import helpers.helpers as h
from model import Model

class LinearRegression(Model):
    def __init__(self, num_inputs: int, alpha: float):
        self.alpha = alpha
        self.w = np.random.rand(num_inputs + 1, 1)

    def train(self, x: np.ndarray, y: np.ndarray, num_iter: int):
        x = h.with_bias(x)
        for _ in range(num_iter):
            grad: np.ndarray = x.T @ ((x @ self.w) - y)
            self.w -= self.alpha / y.shape[0] * grad

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = h.with_bias(x)
        return x @ self.w

if __name__ == "__main__":
    reg = LinearRegression(1, 1e-2)
    x = np.array([[0, 2, 4, 6, 8]]).T
    y = np.array([[1, 4, 7, 10, 13]]).T
    reg.train(x, y, 1000)
    print(reg.evaluate(x))
