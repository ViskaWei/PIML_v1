import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from PIML.crust.data.constants import Constants

class BaseGrid(ABC):
    """ Base class for all spec grids. """
    required_attributes = ["grid_coord", "grid_idx"]
    def set_coord(self, coord):
        self.coord = coord


class StellarGrid(BaseGrid):
    """ Base class for all spec grids. """
    def __init__(self, coord, coord_idx, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.coord = coord
        self.coord_idx = coord_idx
        
        self.PhyShort = Constants.PhyShort
        self.dfcoord = self.set_dfcoord()

    def set_dfcoord(self):
        return pd.DataFrame(self.coord, columns=self.PhyShort)

    def get_coord_idx(self, coord_i):
        return np.where(np.prod(self.coord == coord_i, 1))[0][0]        


