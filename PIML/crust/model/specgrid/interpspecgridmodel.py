from abc import ABC, abstractmethod

from PIML.crust.model.specgrid.basespecgridmodel import BaseSpecGridModel
# from PIML.core.method.interp.baseinterp import RBFInterpModel
from PIML.core.method.interp.interpbuilder import RBFInterpBuilder
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid

class InterpBuilderSpecGridModel(BaseSpecGridModel):
    @abstractmethod
    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        pass

class RBFInterpBuilderSpecGridModel(InterpBuilderSpecGridModel):

    def set_model_param(self, kernel="gaussian", epsilon=0.5):
        self.builder = RBFInterpBuilder(kernel, epsilon)

    def apply(self, coord, value):
        self.builder.build(coord, value)
        def interpolator(eval_coord):
            if eval_coord.ndim == 1:
                return self.builder.interpolator([eval_coord])[0]
            else:
                return self.builder.interpolator(eval_coord)
        return interpolator

    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        interpolator = self.apply(SpecGrid.coordx, SpecGrid.logflux)
        def coord_interpolator(eval_coord, scale=True):
            coordx = SpecGrid.coordx_scaler(eval_coord) if scale else eval_coord
            return interpolator(coordx)            
        SpecGrid.interpolator = coord_interpolator
        SpecGrid.builder = self.builder

    
class PCARBFInterpBuilderSpecGridModel(RBFInterpBuilderSpecGridModel):
    pass

