import os
from abc import ABC, abstractmethod

from PIML.crust.data.specdata.baseboxparam import BoxParam
from PIML.crust.data.specdata.basespec import StellarSpec
from PIML.crust.data.specdata.basesky import StellarSky
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.surface.database.baseloader import H5pyLoader, NpLoader, PickleLoader, ZarrLoader

class BaseLoaderIF(ABC):
    @abstractmethod
    def load(self, path):
        pass


class PathLoaderIF(object):
    def set_path(self, DATA_PATH: str):
        self.DATA_PATH = DATA_PATH
        if DATA_PATH.endswith(".h5"):
            self.loader = H5pyLoader()
        elif DATA_PATH.endswith(".zarr"):
            self.loader = ZarrLoader()
        elif DATA_PATH.endswith(".npy"):
            self.loader = NpLoader()
        elif DATA_PATH.endswith(".pickle"):
            self.loader = PickleLoader()
        else:
            raise NotImplementedError(f"{DATA_PATH} not implemented")

#------------------------------------------------------------------------------
class DirLoaderIF(PathLoaderIF):
    def set_dir(self, DATA_DIR: str, name, ext):
        DATA_PATH = os.path.join(DATA_DIR, name + ext)
        self.set_path(DATA_PATH)

class ParamLoaderIF(PathLoaderIF):
    def set_param(self, PARAMS):
        self.set_path(PARAMS["path"])

#------------------------------------------------------------------------------
class DictLoaderIF(PathLoaderIF):
    def load_dict_args(self):
        self.keys = self.loader.get_keys(self.DATA_PATH)
        return self.loader.load_dict_args(self.DATA_PATH)

    def load_arg(self, arg):
        return self.loader.load_arg(self.DATA_PATH, arg)
    
    def is_arg(self, arg):
        return self.loader.is_arg(self.DATA_PATH, arg)

class ObjectLoaderIF(DirLoaderIF, BaseLoaderIF):
    def load(self, DATA_PATH: str=None):
        if DATA_PATH is not None: self.set_path(DATA_PATH)
        return self.loader.load(self.DATA_PATH)

#------------------------------------------------------------------------------

class SpecGridLoaderIF(ParamLoaderIF, DictLoaderIF):
    """ class for loading Spec Grid (wave, flux, Physical Param for each flux..). """
    def load(self):
        wave = self.load_arg("wave")
        flux = self.load_arg("flux")
        coord = self.load_arg("para")
        coord_idx  = self.load_arg("pdx") if self.is_arg("pdx") else None
        return StellarSpecGrid(wave, flux, coord, coord_idx)

class SpecLoaderIF(ParamLoaderIF, DictLoaderIF):
    """ class for loading Spec. """
    def load(self):
        flux = self.load_arg("flux")
        wave = self.load_arg("wave")
        return StellarSpec(wave, flux)

class GridLoaderIF(ParamLoaderIF, DictLoaderIF):
    """ class for loading box para. """
    def load(self):
        coord = self.load_arg("para")
        cooord_idx  = self.load_arg("pdx") if self.is_arg("pdx") else None
        return StellarGrid(coord, cooord_idx)

class NNTestLoaderIF(ParamLoaderIF, DictLoaderIF):
    """ class for loading NN. """
    def load(self):
        data = self.load_arg("logflux")
        label = self.load_arg("coord")
        return data, label


class WaveSkyLoaderIF(ObjectLoaderIF):
    """ class for loading Sky. """
    def load(self, DATA_PATH: str=None):
        # PATH = "/home/swei20/PIML_v1/test/testdata/wavesky.npy"
        wavesky = super().load(DATA_PATH=DATA_PATH)
        return StellarSky(wavesky)

class SkyLoaderIF(DictLoaderIF):
    def load(self, path, arm, res):
        self.set_path(path)
        name = f"{arm}_R{res}"
        sky = self.load_arg(name)
        return sky