from abc import ABC, abstractmethod
from PIML.surface.database.baseloader import H5pyLoader, ZarrLoader

class BaseLoaderIF(ABC):
    """ Base class for dataIF. """
    def __init__(self, ) -> None:
        super().__init__()
        self.loader = None
        self.path = None
        self.data = None

    def set_data_path(self, DATA_PATH: str):
        self.path = DATA_PATH
        if DATA_PATH.endswith(".h5"):
            self.loader = H5pyLoader()
        elif DATA_PATH.endswith(".zarr"):
            self.loader = ZarrLoader()

    def load_arg(self, arg):
        return self.loader.load_arg(self.path, arg)

    def load_DArgs(self):
        return self.loader.load_DArgs(self.path)

    @abstractmethod
    def load_data(self):
        pass


class FluxLoaderIF(BaseLoaderIF):
    def __init__(self) -> None:
        super().__init__()

    """ class for loading flux. """
    def load_data(self):
        self.data = self.load_arg("flux")

class WaveLoaderIF(BaseLoaderIF):
    """ class for loading wave. """
    def __init__(self) -> None:
        super().__init__()
    
    def load_data(self):
        self.data = self.load_arg("wave")

class ParaLoaderIF(BaseLoaderIF):
    """ class for loading para. """
    def __init__(self) -> None:
        super().__init__()
    
    def load_data(self):
        self.data = self.load_arg("para")

class PdxLoaderIF(BaseLoaderIF):
    """ class for loading pdx. """
    def __init__(self) -> None:
        super().__init__()

    def load_data(self):
        self.data = self.load_arg("pdx")