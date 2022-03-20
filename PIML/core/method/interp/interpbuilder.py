import numpy as np
import logging

from abc import ABC, abstractmethod
from scipy.interpolate import RBFInterpolator
from PIML.gateway.storerIF.basestorerIF import InterpStorerIF

class InterpBuilder(ABC):
    @abstractmethod
    def build(self, coord, value):
        pass

class RBFInterpBuilder(InterpBuilder):
    def __init__(self, kernel="gaussian", epsilon=0.5) -> None:
        self.kernel = kernel
        self.epsilon = epsilon

    def build(self, coord, value):
        # return self.train_interpolator(coord, value)
        logging.info(f"Building RBF with gaussan kernel on data shape {value.shape}")
        self.interpolator = RBFInterpolator(coord, value, kernel=self.kernel, epsilon=self.epsilon)

    def store(self, DATA_DIR, name="interp"):
        self.storer = InterpStorerIF()
        self.storer.set_data_path(DATA_DIR, name)
        self.storer.store(self.interpolator)
        