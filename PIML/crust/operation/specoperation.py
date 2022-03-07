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


class ArmSplitOperation(SplitOperation):
    """ class for splitting data. """
    def __init__(self, arm: str,) -> None:
        wave_rng = Constants.DWs[arm]
        super().__init__(wave_rng)

    def perform(self, data):
        return super().perform(data)

    def perform_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        split_idxs = self.get_split_idxs(Spec.wave)
        Spec.wave = super().split(Spec.wave, split_idxs)
        Spec.flux = super().split(Spec.flux, split_idxs)


class ResolutionOperation(BaseModelOperation, BaseSpecOperation):
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
        self.model.apply_to_spec(Spec)
