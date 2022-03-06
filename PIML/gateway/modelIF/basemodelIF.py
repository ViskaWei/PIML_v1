import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.model.spec.resolutionmodel import AlexResolutionModel, NpResolutionModel


class BaseModelIF(ABC):
    """ Base class for all models. """

    def set_model_data(self, data):
        pass

    @abstractmethod
    def set_model(self):
        pass

    @abstractmethod
    def set_model_param(self, param):
        pass

    @abstractmethod
    def apply_model(self):
        pass


class ResolutionModelIF(BaseModelIF):
    def set_model(self, model_type):
        if model_type == "Alex":
            self.model = AlexResolutionModel()
        elif model_type == "Np":
            self.model = NpResolutionModel()
        else:
            raise ValueError("Unknown Resolution model type: {}".format(model_type))
    
    def set_model_param(self, param):
        self.model.set_model_param(param["step"])

    def apply_model(self, data):
        return self.model.apply(data)
        
        