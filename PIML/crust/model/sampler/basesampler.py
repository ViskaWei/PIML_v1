import numpy as np
from scipy.stats.qmc import Halton
from abc import ABC, abstractmethod

class BaseSamplerBuilder(ABC):
    """ Base class for Sampling. """
    @abstractmethod
    def sample(self, N):
        pass

class Sampler(SamplerBuilder):
    def __init__(self, ndim, rng, method:str="uniform", seed=None):
        assert len(rng) == ndim
        self.rng = rng
        self.ndim = ndim
        self.seed = seed
        self.sampler = self.set_sampler(method)

    def set_sampler(self, method):
        if method == "halton":
            return self.get_halton_sampler()
        elif method == "uniform":
            return self.get_uniform_sampler()
        else:
            raise ValueError("Sampling method is not supported.")

    def sample(self, N):
        return self.sampler(N)

    def get_halton_sampler(self):
        ''' Halton Sampling.
        # Using Halton sequence to generate more evenly spaced samples
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.Halton.html
        '''
        def sampler(N):
            return Halton(d=self.ndim, scramble=False, seed=self.seed).random(n=N)
        return sampler

    def get_uniform_sampler(self):
        def sampler(N):
            if self.seed is not None:
                np.random.seed(self.seed)
            return np.random.uniform(0, self.rng, size=(N, self.ndim))
        return sampler
