import numpy as np
from abc import ABC, abstractmethod

from PIML.crust.model.basemodel import BaseModel
# from PIML.crust.data.spec.basespec import StellarSpec


class BaseSpecModel(BaseModel):
    
    @property
    @abstractmethod
    def name(self):
        return "BaseSpecModel"
    
    @abstractmethod
    def apply(self):
        pass

    @abstractmethod
    def apply_on_Spec(self, Spec):
        pass

    def set_model_param(self, param):
        pass
