import numpy as np
from abc import ABC, abstractmethod
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF
from PIML.gateway.modelIF.basemodelIF import ResolutionModelIF

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
    def set_model(self, model_type):
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

    def set_model(self, model_type):
        self.model = ResolutionModelIF()
        self.model.set_model(model_type)

    def set_param(self, param):
        self.model.set_model_param(param)

    def set_data(self, data):
        self.model.set_model_data(data)

    def process(self):
        return self.model.process()

