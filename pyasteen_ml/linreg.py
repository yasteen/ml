import numpy as np


class LinearRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.shape[1] != y.shape[1] or y.shape[0] != 1:
            raise ValueError("Invalid shape inputs")
        self.x = x
        self.y = y
        self.theta = np.random.rand(x.shape[0] + 1, 1)

    def evaluate(self) -> np.ndarray:
        return self.theta.transpose().dot(self._x_bias())

    def _x_bias(self):
        return np.block([[np.ones((1, self.x.shape[1]))], [self.x]])

    def grad_descent(self, num_iter: int, alpha: float):
        for _ in range(num_iter):
            grad: np.ndarray = self._x_bias().dot(
                (self.evaluate() + -self.y).transpose())
            self.theta -= alpha / self.y.shape[1] * grad


if __name__ == "__main__":
    reg = LinearRegression(
        np.array([[0, 2, 4, 6, 8]]), np.array([[1, 4, 7, 10, 13]]))
    reg.grad_descent(10000, 0.01)
    print(reg.evaluate())
