import numpy as np
import logging
from abc import ABC, abstractmethod
from scipy.interpolate import RBFInterpolator
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
        def interpolator(eval_coord):
            coordx = SpecGrid.coordx_scaler(eval_coord)
            return self.base_interpolator(coordx)            
        SpecGrid.interpolator = interpolator

    #     # coordx = SpecGrid.coordx  if hasattr(SpecGrid, "coordx") else (SpecGrid.coord_idx - SpecGrid.coord_idx[0])
    #     # value  = SpecGrid.logflux if hasattr(SpecGrid, "logflux") else np.log(SpecGrid.flux)
    #     SpecGrid.interpolator = self.apply(SpecGrid.coordx, SpecGrid.logflux)
    
class PCARBFInterpSpecGridModel(RBFInterpSpecGridModel):
    pass


class InterpBuilder(ABC):
    @abstractmethod
    def build(self, coord, value):
        pass

class RBFInterpBuilder(InterpBuilder):
    def __init__(self, kernel="gaussian", epsilon=0.5) -> None:
        self.kernel = kernel
        self.epsilon = epsilon

    def train_interpolator(self, coord, value):
        logging.info(f"Building RBF with gaussan kernel on data shape {value.shape}")
        interpolator = RBFInterpolator(coord, value, kernel=self.kernel, epsilon=self.epsilon)
        return interpolator

    def build(self, coord, value):
        # return self.train_interpolator(coord, value)
        raw_interpolator = self.train_interpolator(coord, value)
        def interpolator(eval_coord):
            if eval_coord.ndim == 1:
                return raw_interpolator(np.array([eval_coord]))[0]
            else:
                return raw_interpolator(eval_coord)
        return interpolator
