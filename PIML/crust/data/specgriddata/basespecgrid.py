from abc import ABC, abstractmethod
import numpy as np
from PIML.crust.data.specdata.basespec import StellarSpec
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.data.constants import Constants

class BaseSpecGrid(ABC):
    def get_grid_bnd(self, coord):
        pass

    def grid_info(self):
        pass

    def print(self):
        pass
        # logging.info(f"#{len(self.wave)} R={self.resolution:.2f}")

class StellarSpecGrid(StellarSpec, StellarGrid, BaseSpecGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_spec == self.coord.shape[0]

    def get_coord_flux(self, coord_i):
        idx = self.get_coord_idx(coord_i)
        return self.flux[idx]
    
    def get_coord_logflux(self, coord_i):
        idx = self.get_coord_idx(coord_i)
        if hasattr(self, 'logflux'):
            return self.logflux[idx]
        else:
            return np.log(self.flux[idx])

