import numpy as np
import logging
from abc import ABC, abstractmethod

class BaseSpec(ABC):
    """ Base class for Spectrum. """
    required_attributes = ["wave", "flux"]

    @abstractmethod
    def get_resolution(self):
        pass

    @abstractmethod
    def num_pixel(self):
        pass

    @abstractmethod
    def num_spec(self):
        pass



class StellarSpec(BaseSpec):
    """ Base class for Stellar Spectrum. """
    def __init__(self, wave, flux, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wave = wave
        self.num_pixel = len(wave)

        assert self.num_pixel == flux.shape[-1]
        self.flux = flux
        self.num_spec = len(flux) if flux.ndim == 2 else 1

        

    def set_wave(self, wave):
        self.wave = wave

    def set_flux(self, flux):
        self.flux = flux
    
    def get_resolution(self):
        resolution = 1 / np.mean(np.diff(np.log(self.wave)))
        logging.info(f"#{len(self.wave)} R={resolution:.2f}")
        return resolution

    def num_pixel(self):
        return self.num_pixel

    def num_spec(self):
        return self.num_spec