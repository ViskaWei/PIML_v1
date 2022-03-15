import numpy as np
from PIML.crust.model.obs.baseobs import Obs

class PfsObs(Obs):
    def set_sky(self, sky):
        self.sky = sky 

    def simulate_sigma(self, flux):
        var = Obs.get_var(flux, self.sky)
        return np.sqrt(var)

    def simulate(self, flux):
        sigma = self.simulate_sigma(flux)
        noise = self.get_noise(sigma)
        return noise + flux


class LowResObs(PfsObs):
    def __init__(self, step):
        self.step = step

    def simulate_sigma(self, fluxL):
        var = super().get_var(fluxL, self.sky)
        var /= self.step
        return np.sqrt(var)
