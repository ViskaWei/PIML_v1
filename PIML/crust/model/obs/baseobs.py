import numpy as np

from abc import ABC, abstractmethod

from PIML.crust.data.spec.basespec import StellarSpec
# from PIML.crust.data.spec.baseobs import StellarObs

class BaseObs(ABC):
    @abstractmethod
    def simulate():
        pass

class Obs(BaseObs):
    @staticmethod
    def get_var(flux, sky):
        #--------------------------------------------
        # Get the total variance
        # BETA is the scaling for the sky
        # VREAD is the variance of the white noise
        # This variance is still scaled with an additional
        # factor when we simuate an observation.
        #--------------------------------------------
        assert flux.shape[-1] == sky.shape[0]
        BETA  = 10.0
        VREAD = 16000
        return  flux + BETA*sky + VREAD

    @staticmethod
    def get_noise(sigma):
        return np.random.normal(0, sigma, np.shape(sigma))

    @staticmethod
    def simulate(flux, sky):
        sigma = Obs.simulate_sigma(flux, sky)
        noise = Obs.get_noise(sigma)
        return noise + flux

    @staticmethod
    def simulate_sigma(flux, sky):
        var = Obs.get_var(flux, sky)
        sigma = np.sqrt(var)
        return sigma

    @staticmethod
    def get_snr(obsfluxs, sigma, noise_level=1):
        return np.mean(np.divide(obsfluxs, noise_level*sigma))

    
