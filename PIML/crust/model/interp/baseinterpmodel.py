import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.model.basemodel import BaseModel

class BaseInterpModel(BaseModel):
    
    @property
    @abstractmethod
    def name(self):
        return "BaseInterpModel"

    @abstractmethod
    def apply(self, data):
        pass

    @abstractmethod
    def train_interpolator(self, coord, value):
        pass
    