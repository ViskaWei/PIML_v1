from abc import ABC, abstractmethod
from PIML.core.method.obs.baseobs import Obs
from PIML.core.method.sampler.gridsampler import StellarGridSampler
from PIML.crust.data.nndata.baseprepnn import BasePrepNN, PrepNN
from PIML.crust.operation.baseoperation import BaseOperation, DataPrepOperation,\
    SamplingOperation, CoordxifyOperation, ObsOperation
# from PIML.crust.operation.samplingoperation import CoordxSamplingOperation

class BasePrepNNOperation(BaseOperation):
    """ Base class for Process of preparing NN data. """
    @abstractmethod
    def perform_on_PrepNN(self, NNP: PrepNN): 
        pass

class CoordxifyPrepNNOperation(CoordxifyOperation, BasePrepNNOperation):
    def __init__(self, coordx_rng) -> None:
        super().__init__(0, coordx_rng)

    def perform_on_PrepNN(self, NNP: PrepNN): 
        self.get_scalers()
        NNP.coordx = self.tick
        NNP.coordx_dim = len(self.tick)
        NNP.label_scaler = self.scaler
        NNP.label_rescaler = self.rescaler

class AddPfsObsNNPredOperation(ObsOperation, BasePrepNNOperation):
    def __init__(self, step) -> None:
        super().__init__(step)
    
    def perform_on_PrepNN(self, NNP: PrepNN):
        Obs = self.perform(NNP.sky)
        NNP.noiser = lambda x: Obs.get_log_sigma(x, log=1)
        NNP.Obs = Obs

class UniformLabelSamplerPrepNNOperation(SamplingOperation, BasePrepNNOperation):
    def __init__(self):
        super().__init__("uniform")

    def perform_on_PrepNN(self, NNP: PrepNN): 
        NNP.uniform_label_sampler = self.perform(NNP.coordx_dim)
        
class HaltonLabelSamplerPrepNNOperation(SamplingOperation, BasePrepNNOperation):
    def __init__(self):
        super().__init__("halton")

    def perform_on_PrepNN(self, NNP: PrepNN): 
        NNP.halton_label_sampler = self.perform(NNP.coordx_dim)

class DataGeneratorPrepNNOperation(BasePrepNNOperation):
    def perform(self, interpolator, rescaler):
        def generator(label):
            return interpolator(rescaler(label))
        return generator

    def perform_on_PrepNN(self, NNP: PrepNN): 
        NNP.generator = self.perform(NNP.interpolator, NNP.label_rescaler)

class TrainPrepNNOperation(DataPrepOperation, BasePrepNNOperation):
    def __init__(self, ntrain) -> None:
        super().__init__(ntrain)
    def perform_on_PrepNN(self, NNP: PrepNN): 
        NNP.train["data"], NNP.train["sigma"], NNP.train["label"] = self.perform(NNP.uniform_label_sampler, NNP.generator, NNP.noiser)

class TestPrepNNOperation(DataPrepOperation, BasePrepNNOperation):
    def __init__(self, ntest) -> None:
        super().__init__(ntest)
    def perform_on_PrepNN(self, NNP: PrepNN):
        NNP.test["data"], NNP.test["sigma"], NNP.test["label"] = self.perform(NNP.halton_label_sampler, NNP.generator, NNP.noiser)
        
