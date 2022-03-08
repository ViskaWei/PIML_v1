import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.model.basemodel import BaseModel


class BaseOperation(ABC):
    """ Base class for Process. """
    @abstractmethod
    def perform(self, data):
        pass

class BaseModelOperation(BaseOperation):
    def __init__(self, model_type, model_param) -> None:
        self.model = self.set_model(model_type)
        self.model.set_model_param(model_param)

    @abstractmethod
    def set_model(self, model_type) -> BaseModel:
        pass

    @abstractmethod
    def perform(self, data):
        super().perform(data)


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
    def __init__(self, rng) -> None:
        self.rng = rng
        self.split_idxs = None

    def get_split_idxs(self, data):
        assert (np.min(data) <= self.rng[0]) and (np.max(data) >= self.rng[1])
        self.split_idxs = np.digitize(self.rng, data)

    def perform(self, data):
        self.get_split_idxs(data)
        return self.split(data, self.split_idxs)

    def split(self, data, split_idxs):
        return data[..., split_idxs[0]:split_idxs[1]]
