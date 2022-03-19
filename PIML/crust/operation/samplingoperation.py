
from PIML.crust.method.sampler.basesampler import SamplerBuilder
from PIML.crust.operation.baseoperation import BaseOperation

class SamplerBuilderOperation(BaseOperation):
    def perform(self, ndim):
        return SamplerBuilder(ndim).build

class SamplerOperation(BaseOperation):
    def __init__(self, sampling_method: str) -> None:
        self.method = sampling_method

    def perform(self, ndim):
        Builder = SamplerBuilder(ndim)
        sampler_fn = Builder.build(self.method)
        return sampler_fn

# class CoordxSamplingOperation(SamplerOperation):
#     def __init__(self, sampling_method: str) -> None:
#         super().__init__(sampling_method)


