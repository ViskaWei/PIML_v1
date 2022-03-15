import numpy as np
from abc import ABC, abstractmethod

from PIML.crust.data.constants import Constants
from PIML.crust.data.spec.basespec import BaseSpec, StellarSpec
from PIML.crust.data.spec.basesky import StellarSky

from PIML.crust.model.obs.basesnrmapper import NoiseLevelSnrMapper
from PIML.crust.model.spec.basespecmodel import BaseSpecModel, AlexResolutionSpecModel, NpResolutionSpecModel
from PIML.crust.operation.baseoperation import BaseOperation, BaseModelOperation, SplitOperation 

class BaseSpecOperation(BaseOperation):
    @abstractmethod
    def perform_on_Spec(self, Spec: StellarSpec):
        pass

class LogSpecOperation(BaseSpecOperation):
    """ Class for taking log of flux """
    def perform(self, flux):
        return np.log(flux)

    def perform_on_Spec(self, Spec: StellarSpec):
        Spec.logflux = self.perform(Spec.flux) 

class SimulateSkySpecOperation(BaseSpecOperation):
    def __init__(self, Sky: StellarSky):
        self.Sky = Sky

    def perform(self, wave):
        return self.Sky.rebin_sky_for_wave(wave)

    def perform_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        Spec.sky = self.perform(Spec.wave)

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
        if hasattr(Spec, "sky"):
            Spec.sky = super().split(Spec.sky, self.split_idxs)

class MapSNRSpecOperation(BaseSpecOperation):
    '''
    perform after sky is simulated with SimulateSkySpecOperation
    '''
    def __init__(self) -> None:
        self.mapper = NoiseLevelSnrMapper()
    
    def perform(self, flux, sky):
        map, map_inv = self.mapper.create_mapper(flux, sky)
        return map, map_inv
    
    def perform_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        Spec.map_snr, Spec.map_snr_inv = self.perform(Spec.flux, Spec.sky)

class TuneSpecOperation(BaseModelOperation, BaseSpecOperation):
    """ class for resolution tunable dataIF i.e flux, wave. """
    def __init__(self, model_type, model_param) -> None:
        super().__init__(model_type, model_param)

    def set_model(self, model_type) -> BaseSpecModel:
        if model_type == "Alex":
            model = AlexResolutionSpecModel()
        elif model_type == "Np":
            model = NpResolutionSpecModel()
        else:
            raise ValueError("Unknown Resolution model type: {}".format(model_type))
        return model

    def perform(self, data):
        return self.model.apply(data)
    
    def perform_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        self.model.apply_on_Spec(Spec)


class SimulateObsSpecOperation(BaseSpecOperation):
    """ class for simulating observation of flux. """
    def __init__(self) -> None:
        self.Obs = StellarObs()        