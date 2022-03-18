from abc import ABC, abstractmethod

from PIML.crust.model.interp.rbfinterp import RBFInterpBuilder
from PIML.crust.model.specgrid.basespecgridmodel import BaseSpecGridModel
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid

class InterpSpecGridModel(BaseSpecGridModel):
    @property
    @abstractmethod
    def name(self):
        return "BaseInterpModel"

class RBFInterpSpecGridModel(InterpSpecGridModel):
    def __init__(self) -> None:
        self.builder = None
        self.base_interpolator = None

    def name(self):
        return "RBF"
        
    def set_model_param(self, kernel="gaussian", epsilon=0.5):
        self.builder = RBFInterpBuilder(kernel, epsilon)

    def set_model_data(self, coord, value):
        self.base_interpolator = self.builder.build(coord, value)

    def apply(self, eval_coord):
        return self.base_interpolator(eval_coord)

    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.set_model_data(SpecGrid.coordx, SpecGrid.logflux)
        def interpolator(eval_coord, scale=True):
            coordx = SpecGrid.coordx_scaler(eval_coord) if scale else eval_coord
            return self.base_interpolator(coordx)            
        SpecGrid.interpolator = interpolator

    
class PCARBFInterpSpecGridModel(RBFInterpSpecGridModel):
    pass

