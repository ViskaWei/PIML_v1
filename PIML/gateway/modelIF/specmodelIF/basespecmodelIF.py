import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
# from PIML.crust.data.spec.basespecgrid import StellarSpecGrid
from PIML.gateway.modelIF.basemodelIF import ResolutionModelIF


class BaseSpecModelIF(ABC):
#  or StellarSpec.__subclasses__()
    def set_spec_model_data(self, spec: StellarSpec):
        self.wave = spec.wave
        self.flux = spec.flux

    @abstractmethod
    def set_spec_model(self, model_type):
        pass
    @abstractmethod
    def set_spec_model_param(self, param):
        pass
    @abstractmethod
    def apply_spec_model(self, Spec: StellarSpec):
        pass


class ResolutionSpecModelIF(BaseSpecModelIF):
    def set_spec_model(self, model_type):
        self.model = ResolutionModelIF()
        self.model.set_model(model_type)

    def set_spec_model_param(self, param):
        self.model.set_model_param(param["step"])

    def apply_spec_model(self,):
        wave = np.exp(self.model.apply_model(np.log(self.wave)))
        flux = self.model.apply_model(self.flux)
        return wave, flux
        
    def apply_on_spec(self, spec: StellarSpec):
        wave, flux = self.apply_spec_model()
        spec.set_wave(wave)
        spec.set_flux(flux)        
        return spec
        


