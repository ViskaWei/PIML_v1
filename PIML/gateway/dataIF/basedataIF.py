import numpy as np
from abc import ABC, abstractmethod

class BaseDataIF(ABC):
    """ Base class for dataIF. """
    
    @abstractmethod
    def set_param(self, param):
        pass

    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def prepare_data(self, ):
        pass




class WaveDataIF(BaseDataIF):
    """ class for separate data into detector arms. """
    def set_param(self, param):
        self.wRng = (param.wStart, param.wEnd)

    def set_data(self, data):
        self.wave = data.wave
        self.flux = data.flux
    
    def prepare_data(self, ):
        np.digitize(self.wRng, self.wave)



    def load_data(self, loader, args):
        pass

class BoxDataIF(BaseDataIF):
    """ class for loading data into box. """
    def __init__(self) -> None:
        super().__init__()

    def set_params(self, boxParams):
        self.R = boxParams["R"]