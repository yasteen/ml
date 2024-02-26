import numpy as np
from helpers.helpers import sigmoid


class LogisticRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.shape[1] != y.shape[1] or y.shape[0] != 1:
            raise ValueError("Invalid shape inputs")
        self.x = x
        self.y = y
        self.theta = np.random.rand(x.shape[0] + 1, 1)

    def hx(self) -> np.ndarray:
        sig = np.vectorize(sigmoid)
        return sig(self.theta.transpose().dot(self._x_bias()))

    def evaluate(self) -> np.ndarray:
        return np.round(self.hx())

    def _x_bias(self):
        return np.block([[np.ones((1, self.x.shape[1]))], [self.x]])

    def grad_descent(self, num_iter: int, alpha: float):
        for _ in range(num_iter):
            grad: np.ndarray = self._x_bias().dot(
                (self.hx() + -self.y).transpose())
            self.theta -= alpha / self.y.shape[1] * grad


if __name__ == "__main__":
    x = np.array([
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
    ])
    y = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    reg = LogisticRegression(
        x, y)
    reg.grad_descent(10000, 0.01)
    print(reg.evaluate())
