
import os
from abc import ABC, abstractmethod
from PIML.surface.database.basestorer import BaseStorer,\
    NpStorer, PickleStorer, H5pyStorer, ZarrStorer

class BaseStorerIF(ABC):
    @abstractmethod
    def store(self):
        pass

class PathStorerIF(BaseStorerIF):
    """ Base class for dataIF. """
    # required_atributes = ["DATA_PATH", "storer"]
    def set_path(self, DATA_PATH: str):
        self.DATA_PATH = DATA_PATH
        if DATA_PATH.endswith(".h5"):
            self.storer = H5pyStorer()
        elif DATA_PATH.endswith(".zarr"):
            self.storer = ZarrStorer()
        elif DATA_PATH.endswith(".npy"):
            self.storer = NpStorer()
        elif DATA_PATH.endswith(".pickle"):
            self.storer = PickleStorer()


class DictStorerIF(PathStorerIF):    
    def store_arg(self, arg, val):
        self.storer.store_arg(self.DATA_PATH, arg, val)

    def store_DArgs(self, DArgs, keys=None):
        if keys is not None:
            DStore = {key: DArgs.__dict__[key] for key in keys}
        else:
            DStore = DArgs
            
        self.storer.store_DArgs(self.DATA_PATH, DStore)

class ObjectStorerIF(PathStorerIF):
    def store(self, PATH, obj):
        self.set_path(PATH)
        self.storer.store(self.DATA_PATH, obj)

class PickleStorerIF(BaseStorerIF):
    def set_path(self, DATA_DIR: str, name):
        self.DATA_DIR = DATA_DIR
        self.DATA_PATH = os.path.join(DATA_DIR, name+".pickle")

    def store(self, data):
        storer = PickleStorer()
        storer.store(self.DATA_PATH, data)

class NNStorerIF(DictStorerIF):
    def store(self, PrepNN):
        def set_path(self, DATA_DIR: str, name):
            self.DATA_DIR = DATA_DIR
            self.DATA_PATH = os.path.join(DATA_DIR, name+".pickle")
        self.store_DArgs(PrepNN.train)