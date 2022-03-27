import numpy as np
from PIML.core.method.obs.baseobs import Obs

class PfsObs(Obs):
    def simulate(self, flux):
        sigma = self.simulate_sigma(flux)
        noise = self.get_noise(sigma)
        return noise + flux
