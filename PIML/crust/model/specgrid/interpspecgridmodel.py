from abc import ABC, abstractmethod

from PIML.crust.model.specgrid.basespecgridmodel import BaseSpecGridModel
# from PIML.crust.method.interp.baseinterp import RBFInterpModel
from PIML.crust.method.interp.interpbuilder import RBFInterpBuilder
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid

class InterpBuilderSpecGridModel(BaseSpecGridModel):
    @abstractmethod
    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        pass


class RBFInterpBuilderSpecGridModel(RBFInterpBuilder, InterpBuilderSpecGridModel):

    def set_model_param(self, kernel="gaussian", epsilon=0.5):
        self.builder = RBFInterpBuilder(kernel, epsilon)

    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        interpolator =self.builder(SpecGrid.coordx, SpecGrid.logflux)
        def interpolator(eval_coord, scale=True):
            coordx = SpecGrid.coordx_scaler(eval_coord) if scale else eval_coord
            return interpolator(coordx)            
        SpecGrid.interpolator = interpolator
        SpecGrid.builder = self.builder

    
class PCARBFInterpBuilderSpecGridModel(RBFInterpBuilderSpecGridModel):
    pass

