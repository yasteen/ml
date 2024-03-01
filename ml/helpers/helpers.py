import math
import numpy as np


def sigmoid(x: float):
    return 1 / (1 + math.e ** (-x))


def sigmoid_grad(a: np.ndarray) -> np.ndarray:
    return a * (1 - a)


def with_bias(x: np.ndarray):
    return np.block([np.ones((x.shape[0], 1)), x])
