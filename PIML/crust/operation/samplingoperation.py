
from PIML.crust.model.sampler.basesampler import SamplerBuilder
from PIML.crust.operation.baseoperation import BaseOperation

class SamplingOperation(BaseOperation):
    def __init__(self, sampling_method: str) -> None:
        self.method = sampling_method

    def perform(self, data):
        pass
        
class CoordxSamplingOperation(SamplingOperation):
    def __init__(self, sampling_method: str) -> None:
        super().__init__(sampling_method)

    def perform(self, coordx_rng):
        ndim = coordx_rng.shape[0]
        rng  = coordx_rng
        offset=0

        builder = SamplerBuilder(ndim, rng, offset=offset)
        sampler = builder.build()
        # sampler(N, seed=None)
        return sampler


