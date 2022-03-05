import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.data.spec.basespec import BaseSpec, StellarSpec
# from PIML.crust.data.spec.basespecgrid import StellarSpecGrid
from PIML.gateway.modelIF.basemodelIF import ResolutionModelIF


class BaseSpecModelIF(ABC):
    """ Base class for model interface for Spec. """
    
    @abstractmethod
    def set_spec_model(self, model_type):
        pass
    @abstractmethod
    def set_spec_model_param(self, param):
        pass
    @abstractmethod
    def set_spec_model_data(self, data: BaseSpec):
        pass
    @abstractmethod
    def apply_model_on_spec(self, Spec: StellarSpec):
        pass


class ResolutionSpecModelIF(BaseSpecModelIF):
    def set_spec_model(self, model_type):
        self.model = ResolutionModelIF()
        self.model.set_model(model_type)

    def set_spec_model_param(self, param):
        self.model.set_model_param(param)

    def set_spec_model_data(self, spec: StellarSpec):
        self.wave = spec.wave
        self.flux = spec.flux

    def apply_spec_model(self, wave, flux):
        wave_new = np.exp(self.model.apply_model(np.log(wave)))
        flux_new = self.model.apply_model(flux)
        return wave_new, flux_new
        
    def apply_model_on_spec(self, spec: StellarSpec):
        wave, flux = self.apply_spec_model(spec.wave, spec.flux)
        spec.set_wave(wave)
        spec.set_flux(flux)        
        return spec
        


