from abc import ABC, abstractmethod
from PIML.crust.data.nndata.basennprep import BaseNNPrep, NNPrep, StellarNNPrep
from PIML.core.method.sampler.gridsampler import StellarGridSampler
from PIML.crust.operation.baseoperation import BaseOperation, SamplerOperation, CoordxifyOperation
# from PIML.crust.operation.samplingoperation import CoordxSamplingOperation


class BaseNNPrepOperation(BaseOperation):
    """ Base class for Process of preparing NN data. """
    @abstractmethod
    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        pass

class SamplerBuilderNNPrepOperation(SamplerOperation, BaseNNPrepOperation):
    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        NNP.sampler_builder = self.perform(NNP.coordx_dim)
        
class CoordxifyNNPrepOperation(CoordxifyOperation, BaseNNPrepOperation):
    def __init__(self, coordx_rng) -> None:
        super().__init__(0, coordx_rng)

    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        self.get_scalers()
        NNP.label_scaler = self.scaler
        NNP.label_rescaler = self.rescaler

class DataGeneratorNNPrepOperation(BaseNNPrepOperation):
    def perform(self, interpolator, rescaler):
        interpolator
        def generator(label):
            coordx = rescaler(label)
            value  = interpolator(coordx)
            return value
        return generator

    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        NNP.data_generator = self.perform(NNP.interpolator, NNP.label_rescaler)

class NzGeneratorNNPrepOperation(BaseNNPrepOperation):
    def perform(self, Obs):
        pass

    def perform_on_NNPrep(self, NNP: StellarNNPrep): 
        NNP.nz_generator = self.perform(NNP.interpolator, NNP.label_rescaler)