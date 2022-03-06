import numpy as np
from abc import ABC, abstractmethod
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF
from PIML.gateway.modelIF.basemodelIF import ResolutionModelIF
from PIML.crust.operation.baseoperation import BaseOperation, SelectOperation, SplitOperation, BoxOperation

class BaseProcessIF(ABC):
    """ Base class for Process interface for data. """
    @abstractmethod
    def process_data(self, data):
        pass

class BaseParamProcessIF(BaseProcessIF):
    """ Base class for Process InterFace for param  """
    @abstractmethod
    def create_process_with_param(self, param):
        pass

class BaseModelProcessIF(BaseProcessIF):
    """ Base class for Process InterFace for model"""
    @abstractmethod
    def set_process_model(self, model_type):
        pass



class SplitProcessIF(BaseParamProcessIF):
    """ class for trimmable dataIF in wavelength direction. i.e. wave, flux, etc. """
    def create_process_with_param(self, param):
        self.process = [SplitOperation(param["startIdx"], param["endIdx"])]
    def process_data(self, data):
        for process in self.process:
            data = process.run(data)

class BoxProcessIF(BaseParamProcessIF):
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

