
import os
from abc import ABC, abstractmethod
from PIML.surface.database.basestorer import BaseStorer,\
    NpStorer, PickleStorer, H5pyStorer, ZarrStorer

class BaseStorerIF(ABC):
    """ Base class for all data loaders. """
    @abstractmethod
    def store(self, path, data):
        pass

class PathStorerIF(object):
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

#------------------------------------------------------------------------------
class DirStoreIF(PathStorerIF):
    def set_dir(self, dir: str, name, ext):
        self.DATA_PATH = os.path.join(dir, name + ext)
        self.set_path(self.DATA_PATH)

class ParamStoreIF(PathStorerIF):
    def set_param(self, PARAMS):
        self.set_path(PARAMS["path"])

#------------------------------------------------------------------------------
class DictStorerIF(DirStoreIF):    
    def store_arg(self, arg, val):
        self.storer.store_arg(self.DATA_PATH, arg, val)

    def store_DArgs(self, DArgs, keys=None):
        if keys is not None:
            DStore = {key: DArgs.__dict__[key] for key in keys}
        else:
            DStore = DArgs
        self.storer.store_DArgs(self.DATA_PATH, DStore)

class ObjectStorerIF(DirStoreIF, BaseStorerIF):
    def store(self, obj, PATH=None):
        if PATH is not None: self.set_path(PATH)
        self.storer.store(self.DATA_PATH, obj)

#------------------------------------------------------------------------------
class InterpStorerIF(ObjectStorerIF):
    def set_dir(self, dir: str, name: str="interp"):
        super().set_dir(dir, name, ".pickle")

# class NNStorerIF(DictStorerIF):
#     def store(self, PrepNN):
#         def set_path(self, DATA_DIR: str, name):
#             self.DATA_DIR = DATA_DIR
#             self.DATA_PATH = os.path.join(DATA_DIR, name+".pickle")
#         self.store_DArgs(PrepNN.train)