import numpy as np
from abc import abstractmethod
from PIML.crust.model.basemodel import BaseModel
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.model.interp.baseinterpmodel import InterpModel, RBFInterpModel, PCARBFInterpModel



class BaseSpecGridmodel(BaseModel):
    @abstractmethod
    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        pass

class InterpSpecGridmodel(InterpModel, BaseSpecGridmodel):

    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        coordx = SpecGrid.coordx  if hasattr(SpecGrid, "coordx") else (SpecGrid.coord_idx - SpecGrid.coord_idx[0])
        value  = SpecGrid.logflux if hasattr(SpecGrid, "logflux") else np.log(SpecGrid.flux)
        SpecGrid.interpolator = self.apply(coordx, value)
        

class RBFInterpSpecGridmodel(RBFInterpModel, InterpSpecGridmodel):

    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        return super().apply_on_SpecGrid(SpecGrid)

class PCARBFInterpSpecGridmodel(PCARBFInterpModel, RBFInterpSpecGridmodel):

    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        return super().apply_on_SpecGrid(SpecGrid)

