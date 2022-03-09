import numpy as np
from abc import ABC, abstractmethod

from PIML.crust.data.constants import Constants
from PIML.crust.data.spec.basespec import StellarSpec
from .baseoperation import BaseOperation, BaseModelOperation, SplitOperation 
from PIML.crust.model.spec.resolutionmodel import ResolutionModel, AlexResolutionModel, NpResolutionModel


class BaseSpecOperation(BaseOperation):
    @abstractmethod
    def perform_on_Spec(self, Spec: StellarSpec):
        pass


class SplitSpecOperation(SplitOperation):
    """ class for splitting data. """
    def __init__(self, arm: str,) -> None:
        wave_rng = Constants.ARM_RNGS[arm]
        super().__init__(wave_rng)

    def perform(self, data):
        return super().perform(data)

    def perform_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        Spec.wave = super().perform(Spec.wave)
        Spec.flux = super().split(Spec.flux, self.split_idxs)



class TuneSpecOperation(BaseModelOperation, BaseSpecOperation):
    """ class for resolution tunable dataIF i.e flux, wave. """
    def __init__(self, model_type, model_param) -> None:
        super().__init__(model_type, model_param)

    def set_model(self, model_type) -> ResolutionModel:
        if model_type == "Alex":
            model = AlexResolutionModel()
        elif model_type == "Np":
            model = NpResolutionModel()
        else:
            raise ValueError("Unknown Resolution model type: {}".format(model_type))
        return model

    def perform(self, data):
        return self.model.apply(data)
    
    def perform_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        self.model.apply_on_Spec(Spec)
