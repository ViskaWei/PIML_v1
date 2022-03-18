import numpy as np
import logging

from abc import ABC, abstractmethod
from scipy.interpolate import RBFInterpolator

class InterpBuilder(ABC):
    @abstractmethod
    def build(self, coord, value):
        pass

class RBFInterpBuilder(InterpBuilder):
    def __init__(self, kernel="gaussian", epsilon=0.5) -> None:
        self.kernel = kernel
        self.epsilon = epsilon

    def train_interpolator(self, coord, value):
        logging.info(f"Building RBF with gaussan kernel on data shape {value.shape}")
        interpolator = RBFInterpolator(coord, value, kernel=self.kernel, epsilon=self.epsilon)
        return interpolator

    def build(self, coord, value):
        # return self.train_interpolator(coord, value)
        raw_interpolator = self.train_interpolator(coord, value)
        def interpolator(eval_coord):
            if eval_coord.ndim == 1:
                return raw_interpolator(np.array([eval_coord]))[0]
            else:
                return raw_interpolator(eval_coord)
        return interpolator