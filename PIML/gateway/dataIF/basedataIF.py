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
    def process_data(self, ):
        pass

    @abstractmethod
    def prepare(self, param, data):
        pass

class WaveDataIF(BaseDataIF):
    """ class for separate data into detector arms. """
    def __init__(self) -> None:
        super().__init__()
        self.startIdx = None
        self.endIdx = None

    def set_param(self, param):
        self.wRng = (param.wStart, param.wEnd)

    def set_data(self, data):
        self.wave = data.wave
    
    def process_data(self):
        self.startIdx, self.endIdx = np.digitize(self.wRng, self.wave)
    
    def prepare(self, param, data):
        self.set_param(param)
        self.set_data(data)
        self.process_data()


class BoxDataIF(BaseDataIF):
    """ class for loading data into box. """
    def __init__(self) -> None:
        super().__init__()

    def set_param(self, param):
        self.R = param.R

    def set_data(self, data):
        self.para = data.para

    def process_data(self):
        dfpara = pd.DataFrame(para, columns=["M","T","G","C","O"])
    
    def prepare(self, param, data):
