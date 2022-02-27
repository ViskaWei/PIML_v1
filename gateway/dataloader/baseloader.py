import os
import h5py
import zarr
import logging
from abc import ABC, abstractmethod


class baseLoader(ABC):
    @abstractmethod
    def load_arg(PATH, arg):
        logging.info(f"Loading {arg} from {PATH}")
        pass

    @staticmethod
    def get_arg(f, arg):
        if baseLoader.is_arg(f, arg):
            return f[arg][:]
        else:
            raise KeyError(f"{arg} not in file")

    @staticmethod
    def is_arg(f, arg):
        return arg in f.keys()


class h5pyLoader(baseLoader):
    def load_arg(self, PATH, arg):
        with h5py.File(PATH, 'r') as f:
            return self.get_arg(f, arg)

class zarrLoader(baseLoader):
    def load_arg(self, PATH, arg):  
        with zarr.open(PATH, 'r') as f:
            return self.get_arg(f, arg)
            

