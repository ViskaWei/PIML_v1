import os
import numpy as np
import h5py
import zarr
import logging
from abc import ABC, abstractmethod


class baseLoader(ABC):
    """ Base class for all data loaders. """

    @abstractmethod
    def load_arg(self, PATH, arg):
        pass

    @abstractmethod
    def load_DArgs(self, PATH):
        pass

    @staticmethod
    def get_arg(f, arg):
        if baseLoader.is_arg(f, arg):
            return f[arg][:]
        else:
            raise KeyError(f"{arg} not in file")

    @staticmethod
    def get_args(f):
        DArgvals = {}
        for arg in f.keys():
            DArgvals[arg] = f[arg][:] 
        return DArgvals

    @staticmethod
    def is_arg(f, arg):
        return arg in f.keys()


class h5pyLoader(baseLoader):
    def load_arg(self, PATH, arg):
        with h5py.File(PATH, 'r') as f:
            logging.info(f"h5pyLoading {arg} from {PATH}")
            return baseLoader.get_arg(f, arg)

    def load_DArgs(self, PATH):
        with h5py.File(PATH, 'r') as f:
            logging.info(f"h5pyLoading {f.keys} from {PATH}")
            return baseLoader.get_args(f)

class zarrLoader(baseLoader):
    def load_arg(self, PATH, arg):  
        with zarr.open(PATH, 'r') as f:
            return baseLoader.get_arg(f, arg)
            
    def load_DArgs(self, PATH):
        with zarr.open(PATH, 'r') as f:
            return baseLoader.get_args(f)