import numpy as np
import logging
from abc import ABC, abstractmethod

class BaseOperation(ABC):
    """ Base class for Process. """
    @abstractmethod
    def run(self, data):
        pass

class SelectOperation(BaseOperation):
    """ class for selective process. """

    def __init__(self, IdxSelected) -> None:
        self.IdxSelected = IdxSelected

    def run(self, data):
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

    def set_operation_param(self, param):
        self.startIdx = param["startIdx"]
        self.endIdx = param["endIdx"]

    def run(self, data):
        return data[..., self.startIdx:self.endIdx]

class ResOperation(BaseOperation):
    """ class for resolution tunable dataIF i.e flux, wave. """

    def set_operation_param(self, param):
        self.step = param["step"]

    def run(self, data):
        