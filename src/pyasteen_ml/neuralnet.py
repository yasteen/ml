import math
from typing import Callable, List
import numpy as np


def sigmoid(x: float):
    return 1 / (1 + math.e ** (-x))


def sigmoid_grad(x: float):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid_grad_simple(o: np.ndarray) -> np.ndarray:
    return o * (1 - o)


class NeuralNetwork:
    """ A neural network consisting of layers """

    def __init__(self,
                 layers: np.ndarray,
                 learn_rate: float,
                 act_func: Callable[[float], float] = sigmoid,
                 act_deriv: Callable[[np.ndarray],
                                     np.ndarray] = sigmoid_grad_simple
                 ):
        NeuralNetwork._verify_init_input(layers)
        self.thetas: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            self.thetas.append(np.random.rand(
                layers[i + 1][0], layers[i][0] + 1))

        self.learn_rate = learn_rate
        self.activation = np.vectorize(act_func)
        self.act_deriv = act_deriv

    def propagate(self, input: np.ndarray) -> List[np.ndarray]:
        self._verify_propagate_input(input)
        a = [input]
        for theta in self.thetas:
            a_bias = np.append(np.ones((1, input.shape[1])), a[-1], 0)
            a.append(self.activation(np.dot(theta, a_bias)))
        return a

    def back_propagate(self, y: np.ndarray, a: List[np.ndarray]) -> List[np.ndarray]:
        deltas: List[np.ndarray] = [np.array([])] * len(a)
        delts: List[np.ndarray] = [np.array([])] * len(self.thetas)

        deltas[-1] = a[-1] + - y

        for i in reversed(range(len(a) - 1)):
            a_bias = np.append(np.ones((1, y.shape[1])), a[i], 0)
            deltas[i] = (self.thetas[i].transpose().dot(
                deltas[i + 1]) * self.act_deriv(a_bias))[1:]
            delts[i] = deltas[i + 1].dot(a_bias.transpose()) / y.shape[1]
        return delts

    def update_weights(self, delts: List[np.ndarray]):
        for i in range(len(self.thetas)):
            self.thetas[i] = self.thetas[i] - delts[i] * self.learn_rate

    def _verify_propagate_input(self, input: np.ndarray):
        if input.shape[0] != self.thetas[0].shape[1] - 1:
            raise ValueError(
                f"Input has unexpected shape. Shape: {input.shape}, expected: {(self.thetas[0].shape[1] - 1, 'm')}")

    @staticmethod
    def _verify_init_input(layers: np.ndarray):
        if len(layers.shape) != 2 or layers.shape[1] != 1:
            raise ValueError("layers array should be n x 1")
        if layers.shape[0] < 2:
            raise ValueError("Length of layers object should be at least 2")
        if np.any(layers < 1):
            raise ValueError(
                "Number of nodes in each layer should be at least 1")


if __name__ == "__main__":
    n = NeuralNetwork(np.array([[2], [1]]), 0.1)

    for i in range(100000):
        a = n.propagate(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]))
        delts = n.back_propagate(np.array([[0, 1, 1, 1]]), a)
        n.update_weights(delts)

    print(n.propagate(np.array([[0, 0, 1, 1], [0, 1, 0, 1]]))[-1])
