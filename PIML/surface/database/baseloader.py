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
    def load_arg(self, PATH, arg):
        pass

    @abstractmethod
    def load_DArgs(self, PATH):
        pass

    @staticmethod
    def _get_arg(f, arg):
        if BaseLoader._is_arg(f, arg):
            return f[arg][:]
        else:
            raise KeyError(f"{arg} not in file")

    @staticmethod
    def _get_args(f):
        DArgvals = {}
        for arg in f.keys():
            DArgvals[arg] = f[arg][:] 
        return DArgvals

    @staticmethod
    def _is_arg(f, arg):
        return arg in f.keys()


class H5pyLoader(BaseLoader):
    
    def get_keys(self, PATH):
        with h5py.File(PATH, 'r') as f:
            return f.keys()

    def is_arg(self, PATH, arg):
        with h5py.File(PATH, 'r') as f:
            return BaseLoader._is_arg(f, arg)

    def load_arg(self, PATH, arg):
        with h5py.File(PATH, 'r') as f:
            logging.info(f"h5pyLoading {arg} from {PATH}")
            return BaseLoader._get_arg(f, arg)

    def load_DArgs(self, PATH):
        with h5py.File(PATH, 'r') as f:
            logging.info(f"h5pyLoading {f.keys} from {PATH}")
            return BaseLoader._get_args(f)

class ZarrLoader(BaseLoader):
    def load_arg(self, PATH, arg):  
        with zarr.open(PATH, 'r') as f:
            return BaseLoader._get_arg(f, arg)
            
    def load_DArgs(self, PATH):
        with zarr.open(PATH, 'r') as f:
            return BaseLoader._get_args(f)


class NpLoader(BaseLoader):
    def load_arg(self, PATH):
        return np.load(PATH)

    def load_DArgs(self, PATH):
        raise NotImplementedError("NpLoader does not support DArgs")

    def load_csv(self, PATH, delimiter=','):
        return np.genfromtxt(PATH, delimiter=delimiter)

        # def load_skyOG(self):
        # skyOG = np.genfromtxt(self.DATADIR +'skybg_50_10.csv', delimiter=',')
        # skyOG[:, 0] = 10 * skyOG[:, 0]
        # return skyOG

class PickleLoader(BaseLoader):
    def load_arg(self, PATH):
        with open(PATH, 'rb') as f:
            return pickle.load(f)
        
    def load_DArgs(self, PATH):
        raise NotImplementedError("PickleLoader does not support DArgs")