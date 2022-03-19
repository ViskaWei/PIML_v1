import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.model.basemodel import BaseModel
from PIML.crust.model.sampler.basesampler import SamplerBuilder


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

class CoordxifyOperation(BaseOperation):
    def __init__(self, origin, tick) -> None:
        self.origin = origin
        self.tick = tick

    def get_scalers(self):
        self.scaler = lambda x: (x - self.origin) / self.tick
        self.rescaler = lambda x: x * self.tick + self.origin

    def perform(self, coord):
        self.get_scalers()
        return self.scaler(coord)

# class BuildLaberGeneratorOperation(BaseOperation):
#     def __init__(self, sampler) -> None:
#         self.label_scaler = label_scaler
#         self.label_rescaler = label_rescaler

#     def perform(self, label):
#         return self.label_rescaler(self.label_scaler(label))

# class BuildDataGeneratorOperation(BaseOperation):
#     def __init__(self, ) -> None:
#         self.data_generator = data_generator

#     def perform(self, data):
#         return self.data_generator(data)

class ApplyInterpOperation(BaseOperation):
    def __init__(self, interpolator, rescaler=None) -> None:
        self.interpolator = interpolator
        self.rescaler = rescaler

    def perform(self, data):
        coordx = data if self.rescaler is None else self.rescaler(data)
        return self.interpolator(coordx)