import numpy as np
from abc import ABC, abstractmethod

from tables import test
from PIML.core.method.obs.baseobs import Obs
from PIML.core.method.sampler.samplerbuilder import SamplerBuilder
from PIML.crust.model.basemodel import BaseModel


class BaseOperation(ABC):
    """ Base class for Process. """
    @abstractmethod
    def perform(self, data):
        pass

class BaseModelOperation(BaseOperation):
    def __init__(self, MODEL) -> None:
        self.model: BaseModel = self.set_model(MODEL["type"])
        self.model.set_model_param(MODEL["param"])

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

class ApplyInterpOperation(BaseOperation):
    def __init__(self, interpolator, rescaler=None) -> None:
        self.interpolator = interpolator
        self.rescaler = rescaler

    def perform(self, data):
        coordx = data if self.rescaler is None else self.rescaler(data)
        return self.interpolator(coordx)

class SamplingOperation(BaseOperation):
    def __init__(self, method):
        self.method = method
    
    def perform(self, ndim):
        builder = SamplerBuilder(ndim)
        sampler = builder.build(self.method)
        return sampler

class ObsOperation(BaseOperation):
    def __init__(self, step) -> None:
        self.step = step
    
    def perform(self, sky):
        if (self.step is None) or (self.step<1): self.step = 1
        return Obs(sky, step=self.step) 

class LabelPrepOperation(BaseOperation):
    def __init__(self, ntrain, ntest, seed=None) -> None:
        self.ntrain = ntrain
        self.ntest = ntest
        self.seed = seed

    def perform(self, train_sampler, test_sampler=None):
        if test_sampler is None: test_sampler = train_sampler
        train_label = train_sampler(self.ntrain, seed=self.seed)
        test_label  = test_sampler(self.ntest, seed=self.seed)
        return train_label, test_label
        
class DataPrepOperation(BaseOperation):

    def perform(self, label, rescaler, interpolator, noiser):
        coordx = rescaler(label)
        data  = interpolator(coordx)
        sigma = noiser(data)
        return data, sigma