import os
import h5py
import zarr
import logging
from abc import ABC, abstractmethod

class baseStorer(ABC):
    """ Base class for all data loaders. """

    @abstractmethod
    def store_arg(self, PATH, arg, val):
        pass

    @abstractmethod
    def store_DArgs(self, PATH, DArgs):
        pass

    @staticmethod
    def is_arg(f, arg):
        return arg in f.keys()


class h5pyStorer(baseStorer):
    def store_arg(self, PATH, arg, val):
        with h5py.File(PATH, 'a') as f:
            f.create_dataset(arg, data=val, shape=val.shape)

    def store_DArgs(self, PATH, DArgs):
        with h5py.File(PATH, 'w') as f:    
            for arg, val in DArgs.items():
                f.create_dataset(arg, data=val, shape=val.shape)
        
class zarrStorer(baseStorer):
    def store_arg(self, PATH, arg, val):
        with zarr.open(PATH, 'a') as f:
            pass
    
    def store_DArgs(self, PATH, DArgs):
        with zarr.open(PATH, 'w') as f:
            pass
