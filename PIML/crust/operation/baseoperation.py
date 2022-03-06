import numpy as np
import logging
from abc import ABC, abstractmethod
from PIML.crust.model.basemodel import BaseModel
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.model.spec.resolutionmodel import AlexResolutionModel, NpResolutionModel


class BaseOperation(ABC):
    """ Base class for Process. """
    @abstractmethod
    def perform(self, data):
        pass

class BaseModelOperation(BaseOperation):
    def __init__(self, model_type, model_param) -> None:
        self.model = self.set_model(model_type)
        self.model.set_model_param(model_param)

    def set_model(self, model_type) -> BaseModel:
        pass

class BaseSpecOperation(BaseOperation):

    def perform_on_spec(self, spec: StellarSpec):
        pass

class ResOperation(BaseModelOperation, BaseSpecOperation):
    """ class for resolution tunable dataIF i.e flux, wave. """

    def set_model(self, model_type):
        if model_type == "Alex":
            model = AlexResolutionModel()
        elif model_type == "Np":
            model = NpResolutionModel()
        else:
            raise ValueError("Unknown Resolution model type: {}".format(model_type))
        return model

    def perform(self, data):
        return self.model.apply(data)
    
    def perform_on_spec(self, spec: StellarSpec) -> StellarSpec:
        self.model.apply_to_spec(spec)



class SelectOperation(BaseOperation):
    """ class for selective process. """

    def __init__(self, IdxSelected) -> None:
        self.IdxSelected = IdxSelected

    def perform(self, data):
        return data[self.IdxSelected, ...]

# TODO FIXME:
class BoxOperation(SelectOperation):
#     """ class for boxable data i.e flux, parameter, etc. """
    pass
#     def __init__(self, IdxInBox) -> None:
#         super().__init__()
    
#     def set_process(self, param):
#         self.IdxInBox = param["IdxInBox"]
    
#     def process_data(self, data):
#         return super().process_data(data)
    

class SplitOperation(BaseOperation):
    """ class for splitting data. """
    def __init__(self, startIdx, endIdx) -> None:
        self.startIdx = startIdx
        self.endIdx = endIdx

    def perform(self, data):
        return data[..., self.startIdx:self.endIdx]

        