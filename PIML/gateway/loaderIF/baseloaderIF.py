from abc import ABC, abstractmethod
from PIML.crust.data.spec.baseboxparam import BoxParam
from PIML.crust.data.spec.basespec import Spec
from PIML.surface.database.baseloader import H5pyLoader, ZarrLoader

class BaseLoaderIF(ABC):
    """ Base class for dataIF. """
    # required_atributes = ["DATA_PATH", "loader"]

    def set_data_path(self, DATA_PATH: str):
        self.DATA_PATH = DATA_PATH
        if DATA_PATH.endswith(".h5"):
            self.loader = H5pyLoader()
        elif DATA_PATH.endswith(".zarr"):
            self.loader = ZarrLoader()

    def load_arg(self, arg):
        return self.loader.load_arg(self.DATA_PATH, arg)

    def load_DArgs(self):
        return self.loader.load_DArgs(self.DATA_PATH)
    
    def is_arg(self, arg):
        return self.loader.is_arg(self.DATA_PATH, arg)

    @abstractmethod
    def load(self):
        pass

class SpecLoaderIF(BaseLoaderIF):
    """ class for loading Spec. """
    def load(self):
        flux = self.load_arg("flux")
        wave = self.load_arg("wave")
        return Spec(wave, flux)

class FluxLoaderIF(BaseLoaderIF):
    """ class for loading flux. """
    def load(self):
        return self.load_arg("flux")

class WaveLoaderIF(BaseLoaderIF):
    """ class for loading wave. """
    def load(self):
        return self.load_arg("wave")

class BoxParamLoaderIF(BaseLoaderIF):
    """ class for loading box para. """
    def load(self):
        para = self.load_arg("para")
        pdx  = self.load_arg("pdx") if self.is_arg("pdx") else None
        return BoxParam(para, pdx=pdx)
