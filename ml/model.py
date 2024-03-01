from abc import abstractmethod
import numpy as np

class Model:
    @abstractmethod
    def train(self, x: np.ndarray, y: np.ndarray):
        return NotImplemented

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return NotImplemented

