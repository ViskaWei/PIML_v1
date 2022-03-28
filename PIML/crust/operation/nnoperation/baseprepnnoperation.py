import numpy as np

from abc import ABC, abstractmethod
from PIML.core.method.obs.baseobs import Obs
from PIML.core.method.sampler.gridsampler import StellarGridSampler
from PIML.crust.data.nndata.baseprepnn import BasePrepNN, PrepNN
from PIML.crust.operation.baseoperation import BaseOperation, DataPrepOperation,\
    SamplingOperation, CoordxifyOperation, ObsOperation,\
    LabelPrepOperation
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

class LabelPrepNNOperation(LabelPrepOperation, BasePrepNNOperation):
    def __init__(self, ntrain, ntest, seed=None) -> None:
        super().__init__(ntrain, ntest, seed)

    def perform_on_PrepNN(self, NNP: PrepNN): 
        NNP.ntrain = self.ntrain
        NNP.ntest = self.ntest
        NNP.seed = self.seed
        NNP.train["label"], NNP.test["label"] = self.perform(NNP.uniform_label_sampler, NNP.halton_label_sampler)

class DataPrepNNOperation(DataPrepOperation, BasePrepNNOperation):

    def perform_on_PrepNN(self, NNP: PrepNN): 
        label = np.vstack((NNP.train["label"], NNP.test["label"]))
        data, sigma = self.perform(label, NNP.label_rescaler, NNP.interpolator, NNP.noiser)
        NNP.train["data"] , NNP.test["data"]  = data [:NNP.ntrain], data [NNP.ntrain:]
        NNP.train["sigma"], NNP.test["sigma"] = sigma[:NNP.ntrain], sigma[NNP.ntrain:]

class FinishPrepNNOperation(BasePrepNNOperation):
    def perform(self, name, n, suffix=""):
        return f"{name}_N{n}{suffix}"

    def perform_on_PrepNN(self, NNP: PrepNN):
        NNP.train_name = self.perform(NNP.name, NNP.ntrain, suffix="_train")
        NNP.test_name = self.perform(NNP.name, NNP.ntest, suffix="_test")

        