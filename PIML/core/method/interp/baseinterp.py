import numpy as np
from abc import ABC, abstractmethod

class BaseInterp(ABC):
    @abstractmethod
    def interp(self):
        pass

class RBFInterp(BaseInterp):
    def __init__(self, interpolator) -> None:
        self.interpolator = interpolator

    def interp(self, eval_coord):
        return self.interpolator(eval_coord)