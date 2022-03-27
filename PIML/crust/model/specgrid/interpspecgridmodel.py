from abc import ABC, abstractmethod

from PIML.crust.model.specgrid.basespecgridmodel import BaseSpecGridModel
from PIML.crust.model.interpmodel import RBFInterpBuilderModel
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid

class InterpBuilderSpecGridModel(BaseSpecGridModel):
    @abstractmethod
    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        pass

class RBFInterpBuilderSpecGridModel(InterpBuilderSpecGridModel):

    def set_model_param(self, kernel="gaussian", epsilon=0.5):
        self.builder = RBFInterpBuilderModel(kernel, epsilon)

    def apply_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        interpolator = self.builder.apply(SpecGrid.coordx, SpecGrid.logflux)
        def coord_interpolator(eval_coord, scale=True):
            coordx = SpecGrid.coordx_scaler(eval_coord) if scale else eval_coord
            return interpolator(coordx)            
        SpecGrid.interpolator = coord_interpolator
        SpecGrid.builder = self.builder

    
class PCARBFInterpBuilderSpecGridModel(RBFInterpBuilderSpecGridModel):
    pass

