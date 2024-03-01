import numpy as np
import helpers.helpers as h
from model import Model

class LogisticRegression(Model):
    def __init__(self, num_inputs: int, alpha: float):
        self.alpha = alpha
        self.w = np.random.rand(num_inputs + 1, 1)

    def train(self, x: np.ndarray, y: np.ndarray, num_iter: int):
        x = h.with_bias(x)
        for _ in range(num_iter):
            grad: np.ndarray = x.T @ (self._evaluate(x) - y)
            self.w -= self.alpha / y.shape[0] * grad

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        sig = np.vectorize(h.sigmoid)
        return sig(x @ self.w)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = h.with_bias(x)
        return np.round(self._evaluate(x))

if __name__ == "__main__":
    x = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    ]).T
    y = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]).T
    reg = LogisticRegression(2, 1e-2)
    reg.train(x, y, 10000)
    print(reg.evaluate(x))
