
from PIML.crust.model.sampler.basesampler import Sampling
from PIML.crust.operation.baseoperation import BaseOperation, BaseModelOperation, SplitOperation 

class SamplingOperation(BaseOperation):
    def __init__(self, sampling_method: str) -> None:
        self.method = sampling_method

    def perform(self, coordx):
        ndim = coordx.shape[1]
        rng  = coordx.max(0) - coordx.min(0)
        self.Sampling = Sampling(ndim, rng, method=self.method)
        
        pass