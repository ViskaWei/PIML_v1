import os
import numpy as np
import pandas as pd
import h5py
import zarr
import pickle
import logging
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """ Base class for all data loaders. """
    @abstractmethod
    def load(self, path) -> object:
        pass

class NpLoader(BaseLoader):
    def load(self, PATH):
        return np.load(PATH)

    def load_csv(self, PATH, delimiter=','):
        return np.genfromtxt(PATH, delimiter=delimiter)

class PickleLoader(BaseLoader):
    def load(self, PATH):
        with open(PATH, 'rb') as f:
            return pickle.load(f)


class BaseDictLoader(ABC):
    """ Base class for all data loaders. """
    @abstractmethod
    def load_arg(self, PATH, arg):
        pass

    @abstractmethod
    def load_dict_args(self, PATH):
        pass

class DictLoader(BaseDictLoader):
    @staticmethod
    def _is_arg(f, arg):
        return arg in f.keys()

    @staticmethod
    def _get_arg(f, arg):
        if DictLoader._is_arg(f, arg):
            return f[arg][:]
        else:
            raise KeyError(f"{arg} not in file")

    @staticmethod
    def _get_args(f):
        DArgvals = {}
        for arg in f.keys():
            DArgvals[arg] = f[arg][:] 
        return DArgvals

class H5pyLoader(DictLoader):
    
    def get_keys(self, PATH):
        with h5py.File(PATH, 'r') as f:
            return f.keys()

    def is_arg(self, PATH, arg):
        with h5py.File(PATH, 'r') as f:
            return self._is_arg(f, arg)

    def load_arg(self, PATH, arg):
        with h5py.File(PATH, 'r') as f:
            logging.info(f"h5pyLoading {arg} from {PATH}")
            return self._get_arg(f, arg)

    def load_dict_args(self, PATH):
        with h5py.File(PATH, 'r') as f:
            logging.info(f"h5pyLoading {f.keys} from {PATH}")
            return self._get_args(f)

class ZarrLoader(DictLoader):
    def load_arg(self, PATH, arg):  
        with zarr.open(PATH, 'r') as f:
            return self._get_arg(f, arg)
            
    def load_dict_args(self, PATH):
        with zarr.open(PATH, 'r') as f:
            return self._get_args(f)


