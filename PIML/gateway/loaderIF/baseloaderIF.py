from abc import ABC, abstractmethod
from PIML.crust.data.spec.baseboxparam import BoxParam
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid

from PIML.surface.database.baseloader import H5pyLoader, ZarrLoader


class BaseLoaderIF(ABC):
    @abstractmethod
    def load(self):
        pass

class ObjectLoaderIF(BaseLoaderIF):
    """ Base class for dataIF. """
    # required_atributes = ["DATA_PATH", "loader"]

    def set_data_path(self, DATA_PATH: str):
        self.DATA_PATH = DATA_PATH
        if DATA_PATH.endswith(".h5"):
            self.loader = H5pyLoader()
        elif DATA_PATH.endswith(".zarr"):
            self.loader = ZarrLoader()

        self.keys = self.loader.get_keys(self.DATA_PATH)

    def load_arg(self, arg):
        return self.loader.load_arg(self.DATA_PATH, arg)

    def load_DArgs(self):
        return self.loader.load_DArgs(self.DATA_PATH)
    
    def is_arg(self, arg):
        return self.loader.is_arg(self.DATA_PATH, arg)

    def load(self):
        pass

class SpecGridLoaderIF(ObjectLoaderIF):
    """ class for loading Spec Grid (wave, flux, Physical Param for each flux..). """
    def load(self):
        wave = self.load_arg("wave")
        flux = self.load_arg("flux")
        coord = self.load_arg("para")
        coord_idx  = self.load_arg("pdx") if self.is_arg("pdx") else None
        return StellarSpecGrid(wave, flux, coord, coord_idx)

class SpecLoaderIF(ObjectLoaderIF):
    """ class for loading Spec. """
    def load(self):
        flux = self.load_arg("flux")
        wave = self.load_arg("wave")
        return StellarSpec(wave, flux)

class GridLoaderIF(ObjectLoaderIF):
    """ class for loading box para. """
    def load(self):
        coord = self.load_arg("para")
        cooord_idx  = self.load_arg("pdx") if self.is_arg("pdx") else None
        return StellarGrid(coord, cooord_idx)
