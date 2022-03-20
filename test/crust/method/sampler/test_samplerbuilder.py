

import numpy as np
from scipy.stats.qmc import Halton

from unittest import TestCase
from PIML.crust.method.sampler.samplerbuilder import SamplerBuilder

class TestBaseSampler(TestCase):

    def test_SamplerBuilder(self):
        ndim, N, seed = 3, 2, 922
        builder = SamplerBuilder(ndim=ndim)
        sampler_fn = builder.build("uniform")
        samples = sampler_fn(N, seed=seed)

        samples_to_check = self.check_uniform_sampler(ndim, N, seed)
        self.assertIsNone(np.testing.assert_array_equal(samples, samples_to_check))

        sampler_fn = builder.build("halton")
        samples = sampler_fn(N, seed=seed)
        samples_to_check = self.check_halton_sampler(ndim, N, seed)
        self.assertIsNone(np.testing.assert_array_equal(samples, samples_to_check))
        
    def check_uniform_sampler(self, ndim, N, seed):
        np.random.seed(seed)
        return np.random.uniform(0, 1, size=(N, ndim))

    def check_halton_sampler(self, ndim, N, seed):
        return Halton(d=ndim, scramble=False, seed=seed).random(n=N)