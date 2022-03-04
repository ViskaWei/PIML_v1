import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.data.baseprocess import ResTunableProcess, AlexResTunableProcess, NpResTunableProcess

from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF

class BaseProcessIF(ABC):
    """ Base class for Process. """
    
    @abstractmethod
    def set_param(self, param):
        pass
    
    @abstractmethod
    def set_data(self, data):
        pass

    @abstractmethod
    def process(self):
        pass


class BaseParamProcessIF(BaseProcessIF):
    """ Base class for Process InterFace between . """
    pass    


class BaseModelProcessIF(BaseProcessIF):
    """ Base class for Process InterFace between . """

    @abstractmethod
    def set_model_type(self, model_type):
        pass


class TrimmableProcessIF(BaseParamProcessIF):
    """ class for trimmable dataIF in wavelength direction. i.e. wave, flux, etc. """
    def __init__(self) -> None:
        super().__init__()
        self.startIdx = None
        self.endIdx = None

    def set_param(self, param):
        self.startIdx = param["startIdx"]
        self.endIdx = param["endIdx"]

    def set_data(self, data):
        self.data = data

    def process(self):
        return self.data[..., self.startIdx:self.endIdx]

class BoxableProcessIF(BaseParamProcessIF):
    """ class for boxable data i.e flux, parameter, etc. """
    def __init__(self) -> None:
        super().__init__()
        self.IdxInBox = None

    def set_param(self, param):
        self.IdxInBox = param["IdxInBox"]
    
    def set_data(self, data):
        self.data = data
    
    def process(self):
        return self.data[self.IdxInBox, ...]
    
class ResTunableProcessIF(BaseParamProcessIF, BaseModelProcessIF):
    """ class for resolution tunable dataIF i.e flux, wave. """
    def __init__(self, ) -> None:
        super().__init__()
    
    def set_model_type(self, MODE):
        self.name = MODE["ResTunable_model_type"]
        if self.name == "Alex":
            self.model = AlexResTunableProcess()
        elif self.name == "Np":
            self.model = NpResTunableProcess()
        else:
            raise ValueError(f"ResTunableMode {self.mode} is not supported.")

    def set_param(self, param):
        self.model.set_process_param(param)

    def set_data(self, data):
        self.model.set_process_data(data)

    def process(self):
        return self.model.process()

