from abc import ABC, abstractmethod
from PIML.core.method.obs.baseobs import Obs
from PIML.core.method.sampler.gridsampler import StellarGridSampler
from PIML.crust.data.nndata.basennprep import BaseNNPrep, NNPrep
from PIML.crust.operation.baseoperation import BaseOperation,\
    SamplingOperation, CoordxifyOperation, ObsOperation
# from PIML.crust.operation.samplingoperation import CoordxSamplingOperation

class BaseNNPrepOperation(BaseOperation):
    """ Base class for Process of preparing NN data. """
    @abstractmethod
    def perform_on_NNPrep(self, NNP: NNPrep): 
        pass

class CoordxifyNNPrepOperation(CoordxifyOperation, BaseNNPrepOperation):
    def __init__(self, coordx_rng) -> None:
        super().__init__(0, coordx_rng)

    def perform_on_NNPrep(self, NNP: NNPrep): 
        self.get_scalers()
        NNP.coordx = self.tick
        NNP.coordx_dim = len(self.tick)
        NNP.label_scaler = self.scaler
        NNP.label_rescaler = self.rescaler

class AddPfsObsNNPredOperation(ObsOperation, BaseNNPrepOperation):
    def __init__(self, step) -> None:
        super().__init__(step)
    
    def perform_on_NNPrep(self, NNP: NNPrep):
        NNP.Obs = self.perform(NNP.sky)
        NNP.gen_sigma = Obs.get_log_sigma

class UniformLabelSamplerNNPrepOperation(SamplingOperation, BaseNNPrepOperation):
    def __init__(self):
        super().__init__("uniform")

    def perform_on_NNPrep(self, NNP: NNPrep): 
        NNP.uniform_label_sampler = self.perform(NNP.coordx_dim)
        
class HaltonLabelSamplerNNPrepOperation(SamplingOperation, BaseNNPrepOperation):
    def __init__(self):
        super().__init__("halton")

    def perform_on_NNPrep(self, NNP: NNPrep): 
        NNP.halton_label_sampler = self.perform(NNP.coordx_dim)

class DataGeneratorNNPrepOperation(BaseNNPrepOperation):
    def perform(self, interpolator, rescaler):
        def gen_data_from_label(label):
            return interpolator(rescaler(label))
        return gen_data_from_label

    def perform_on_NNPrep(self, NNP: NNPrep): 
        NNP.gen_data_from_label = self.perform(NNP.interpolator, NNP.label_rescaler)

