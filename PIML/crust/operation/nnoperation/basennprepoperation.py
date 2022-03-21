from abc import ABC, abstractmethod
from PIML.crust.data.nndata.basennprep import BaseNNPrep, NNPrep, StellarNNPrep
from PIML.core.method.sampler.gridsampler import StellarGridSampler
from PIML.crust.operation.baseoperation import BaseOperation, SamplingOperation, CoordxifyOperation
# from PIML.crust.operation.samplingoperation import CoordxSamplingOperation


class BaseNNPrepOperation(BaseOperation):
    """ Base class for Process of preparing NN data. """
    @abstractmethod
    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        pass

class UniformLabelSamplerNNPrepOperation(SamplingOperation, BaseNNPrepOperation):
    def __init__(self):
        super().__init__("uniform")

    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        NNP.uniform_label_sampler = self.perform(NNP.coordx_dim)
        
class HaltonLabelSamplerNNPrepOperation(SamplingOperation, BaseNNPrepOperation):
    def __init__(self):
        super().__init__("halton")

    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        NNP.halton_label_sampler = self.perform(NNP.coordx_dim)

class CoordxifyNNPrepOperation(CoordxifyOperation, BaseNNPrepOperation):
    def __init__(self, coordx_rng) -> None:
        super().__init__(0, coordx_rng)

    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        self.get_scalers()
        NNP.label_scaler = self.scaler
        NNP.label_rescaler = self.rescaler

class DataGeneratorNNPrepOperation(BaseNNPrepOperation):
    def perform(self, interpolator, rescaler):
        def gen_data_from_label(label):
            return interpolator(rescaler(label))
        return gen_data_from_label

    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        NNP.gen_data_from_label = self.perform(NNP.interpolator, NNP.label_rescaler)

class NzGeneratorNNPrepOperation(BaseNNPrepOperation):
    def perform(self, cal_sigma):
        def gen_sigma(data):
            return cal_sigma(data)
        return gen_sigma

    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        NNP.nz_generator = self.perform(NNP.interpolator, NNP.label_rescaler)

