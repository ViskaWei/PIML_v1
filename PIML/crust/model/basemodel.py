import numpy as np
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def set_model_param(self, param):
        pass
    @abstractmethod
    def apply(self):
        pass