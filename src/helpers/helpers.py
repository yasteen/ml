import math
from numpy import ndarray


def sigmoid(x: float):
    return 1 / (1 + math.e ** (-x))


def sigmoid_grad(a: ndarray) -> ndarray:
    return a * (1 - a)
