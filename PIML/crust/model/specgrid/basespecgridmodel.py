import numpy as np
from abc import abstractmethod
from PIML.crust.model.basemodel import BaseModel
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid



class BaseSpecGridModel(BaseModel):
    @abstractmethod
    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        pass
