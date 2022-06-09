import math
from typing import Callable, List
import numpy as np
from ..helpers.helpers import sigmoid, sigmoid_grad


class NeuralNetwork:
    """ A neural network consisting of layers """

    def __init__(self,
                 layers: np.ndarray,
                 learn_rate: float,
                 lam: float = 0,
                 act_func: Callable[[float], float] = sigmoid,
                 act_deriv: Callable[[np.ndarray],
                                     np.ndarray] = sigmoid_grad
                 ):
        NeuralNetwork._verify_init_input(layers)
        self.thetas: List[np.ndarray] = []
        for i in range(len(layers) - 1):
            self.thetas.append(np.random.rand(
                layers[i + 1][0], layers[i][0] + 1))

        self.learn_rate = learn_rate
        self.lam = lam
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

        deltas[-1] = y + -a[-1]

        for i in reversed(range(len(a) - 1)):
            a_bias = np.append(np.ones((1, y.shape[1])), a[i], 0)
            deltas[i] = (self.thetas[i].transpose().dot(
                deltas[i + 1]) * self.act_deriv(a_bias))[1:]
            regularization = np.block(
                [[np.zeros((self.thetas[i].shape[0], 1)), self.thetas[i][:, 1:]]])
            delts[i] = deltas[i + 1].dot(a_bias.transpose()) / y.shape[1] \
                + self.lam / y.shape[1] * regularization
        return delts

    def update_weights(self, delts: List[np.ndarray]):
        for i in range(len(self.thetas)):
            self.thetas[i] = self.thetas[i] + delts[i] * self.learn_rate

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
    # np.random.seed(1)
    x = np.array(
        [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]])
    y = np.array(
        [[0],
         [1],
         [1],
         [0]])
    n = NeuralNetwork(np.array([[2], [2], [1]]), 0.5, 0.00001)

    for i in range(5000):
        a = n.propagate(x.transpose())
        delts = n.back_propagate(y.transpose(), a)
        n.update_weights(delts)

    print(n.propagate(x.transpose())[-1].round(8))
