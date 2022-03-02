import numpy as np
from abc import ABC, abstractmethod

from PIML.gateway.dataIF.baseloaderIF import BaseLoaderIF, WaveLoaderIF

class BasePrepareIF(ABC):
    """ Base class for PrepareIF. """
    def __init__(self) -> None:
        super().__init__()
        self.DATA_PATH = None
    
    @abstractmethod
    def set_param(self, param):
        if "DATA_PATH" in param:
            self.DATA_PATH = param["DATA_PATH"]
        else:
            raise KeyError("No DATA_PATH in param")

    @abstractmethod
    def set_data(self):
        pass

    @abstractmethod
    def process_data(self):
        pass

    @abstractmethod
    def prepare(self, param, data):
        self.set_param(param)
        self.set_data(data)
        self.process_data()

    


class WavePrepareIF(BasePrepareIF, WaveLoaderIF):
    """ class for separate data into detector arms. """
    def __init__(self) -> None:
        super().__init__()
        self.startIdx = None
        self.endIdx = None

    def set_param(self, param):
        super().set_param(param)
        self.wRng = (param.wStart, param.wEnd)

    def set_data(self):
        self.set_param
        self.wave = self.load_data(WaveLoaderIF)

    def process_data(self):
        self.startIdx, self.endIdx = np.digitize(self.wRng, self.wave)
        wave_processed = self.wave[self.startIdx:self.endIdx]
        self.data = wave_processed
    
    def prepare(self, param, data):
        return super().prepare(param, data)



class BoxPrepareIF(BasePrepareIF):
    """ class for loading data into box. """
    def __init__(self) -> None:
        super().__init__()

    def set_param(self, param):
        self.R = param.R

    def set_data(self):
        self.para = data.para

    def process_data(self):
        dfpara = pd.DataFrame(para, columns=["M","T","G","C","O"])
    

    


    def process_data(self):
        self.startIdx, self.endIdx = np.digitize(self.wRng, self.wave)
        wave_processed = self.wave[self.startIdx:self.endIdx]
        self.data = wave_processed
    
    def prepare(self, param, data):
        return super().prepare(param, data)