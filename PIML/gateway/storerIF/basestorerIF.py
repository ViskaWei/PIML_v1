

from abc import ABC, abstractmethod
from PIML.surface.database.basestorer import BaseStorer, H5pyStorer, PickleStorer, ZarrStorer

class BaseStorerIF(ABC):
    @abstractmethod
    def store(self):
        pass

class ObjectStorerIF(BaseStorerIF):
    """ Base class for dataIF. """
    # required_atributes = ["DATA_PATH", "storer"]
    def set_data_path(self, DATA_PATH: str):
        self.DATA_PATH = DATA_PATH
        if DATA_PATH.endswith(".h5"):
            self.storer = H5pyStorer()
        elif DATA_PATH.endswith(".zarr"):
            self.storer = ZarrStorer()

    def store_arg(self, arg, val):
        self.storer.store_arg(self.DATA_PATH, arg, val)

    def store_DArgs(self, DArgs, keys=None):
        if keys is not None:
            DStore = {key: DArgs.__dict__[key] for key in keys}
        else:
            DStore = DArgs
            
        self.storer.store_DArgs(self.DATA_PATH, DStore)

    def store(self):
        pass

class InterpStoreIF(BaseStorerIF):
    def store_arg(self, PATH, interp):
        storer = PickleStorer()
        storer.store_arg(PATH, interp)