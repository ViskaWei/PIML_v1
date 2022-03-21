import numpy as np
from PIML.core.method.obs.baseobs import Obs

class PfsObs(Obs):
        
    def set_sky(self, sky):
        self.sky = sky 

    def cal_sigma(self, flux):
        var = Obs.get_var(flux, self.sky)
        return np.sqrt(var)

    def simulate(self, flux):
        sigma = self.simulate_sigma(flux)
        noise = self.get_noise(sigma)
        return noise + flux


class LowResObs(PfsObs):
    def __init__(self, step):
        self.step = step

    def cal_sigma(self, fluxL):
        var = super().get_var(fluxL, self.sky)
        var /= self.step
        return np.sqrt(var)