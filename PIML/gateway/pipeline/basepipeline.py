
import numpy as np

from abc import ABC, abstractmethod

class BasePipeline(ABC):
    """ Base class for Pipeline. """
    @abstractmethod
    def build(self):
        pass