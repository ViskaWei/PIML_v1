import numpy as np
from abc import ABC, abstractmethod
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF
from PIML.gateway.modelIF.basemodelIF import ResolutionModelIF

class BaseProcessIF(ABC):
    """ Base class for Process interface for data. """
    def set_process_data(self, data):
        pass
    @abstractmethod
    def process(self, data):
        pass

class BaseParamProcessIF(BaseProcessIF):
    """ Base class for Process InterFace for param  """
    @abstractmethod
    def set_process_param(self, param):
        pass

class BaseModelProcessIF(BaseProcessIF):
    """ Base class for Process InterFace for model"""
    @abstractmethod
    def set_process_model(self, model_type):
        pass

class TrimmableProcessIF(BaseParamProcessIF):
    """ class for trimmable dataIF in wavelength direction. i.e. wave, flux, etc. """
    def __init__(self) -> None:
        self.startIdx: int = None
        self.endIdx  : int = None

    def set_process_param(self, param):
        self.startIdx = param["startIdx"]
        self.endIdx = param["endIdx"]

    def process_data(self, data):
        return data[..., self.startIdx:self.endIdx]

    def set_process_data(self, data):
        self.data = data

    def process(self):
        return self.process_data(self.data)

class BoxableProcessIF(BaseParamProcessIF):
    """ class for boxable data i.e flux, parameter, etc. """
    def __init__(self) -> None:
        self.IdxInBox = None

    def set_process_param(self, param):
        self.IdxInBox = param["IdxInBox"]
    
    def process_data(self, data):
        return data[self.IdxInBox, ...]
    def set_process_data(self, data):
        self.data = data
    
    def process(self):
        return self.process_data(self.data)
    
class ResTunableProcessIF(BaseParamProcessIF, BaseModelProcessIF):
    """ class for resolution tunable dataIF i.e flux, wave. """
    def __init__(self) -> None:
        self.modelIF: ResolutionModelIF = None
        self.model_type: str = None
    
    def set_process_model(self, MODEL_TYPES):
        self.modelIF = ResolutionModelIF()
        self.model_type = MODEL_TYPES["ResTunableProcess"]
        self.modelIF.set_model(self.model_type)

    def set_process_param(self, param):
        self.modelIF.set_model_param(param)

    def process_data(self, data):
        return self.modelIF.apply_model(data)

    def set_process_data(self, data):
        self.data = data

    def process(self):
        return self.process_data(self.data)

