import numpy as np
from scipy.stats.qmc import Halton
from abc import ABC, abstractmethod

class BaseSamplerBuilder(ABC):
    """ Base class for Sampling. """
    @abstractmethod
    def build(self, N):
        pass

class SamplerBuilder(BaseSamplerBuilder):
    def __init__(self, ndim):
        self.ndim = ndim

    def build(self, method):
        if method == "halton":
            sampler = self.get_halton_sampler()
        elif method == "uniform":
            sampler = self.get_uniform_sampler()
        else:
            raise ValueError("Sampling method is not supported.")
        return sampler

    def get_halton_sampler(self):
        ''' Halton Sampling.
        # Using Halton sequence to generate more evenly spaced samples
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html
        '''
        def sampler(N, seed=None):
            return Halton(d=self.ndim, scramble=False, seed=seed).random(n=N)
        return sampler

    def get_uniform_sampler(self):
        def sampler(N, seed=None):
            if seed is not None: np.random.seed(seed)
            return np.random.uniform(0, 1, size=(N, self.ndim))
        return sampler
    
