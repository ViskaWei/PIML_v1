from abc import ABC, abstractmethod
from PIML.crust.data.nndata.basennprep import BaseNNPrep, NNPrep, SpecGridNNPrep
from PIML.crust.operation.baseoperation import BaseOperation, SamplerOperation, CoordxifyOperation
# from PIML.crust.operation.samplingoperation import CoordxSamplingOperation


class BaseNNPrepOperation(BaseOperation):
    """ Base class for Process of preparing NN data. """
    @abstractmethod
    def perform_on_NNPrep(self, NNP: SpecGridNNPrep): 
        pass

class SamplerNNPrepOperation(SamplerOperation, BaseNNPrepOperation):
    def perform_on_NNPrep(self, NNP: SpecGridNNPrep): 
        NNP.sampler = self.perform(NNP.coordx_dim)
        
class CoordxifyNNPrepOperation(CoordxifyOperation, BaseNNPrepOperation):
    def __init__(self, coordx_rng) -> None:
        super().__init__(0, coordx_rng)

    def perform_on_NNPrep(self, NNP: SpecGridNNPrep): 
        self.get_scalers()
        NNP.label_scaler = self.scaler
        NNP.label_rescaler = self.rescaler

class BuildLabelMakerNNPrepOperation(BaseNNPrepOperation):
    def 

    def perform_on_NNPrep(self, NNP: SpecGridNNPrep): 
        NNP.label = self.perform(NNP.label)

class MakeDataNNPrepOperation(BaseNNPrepOperation):
    def __init__(self) -> None:
        


    def perform_on_NNPrep(self, NNP: SpecGridNNPrep): 
        NNP.train_idxs = self.perform(NNP.sampler)
        NNP.train_idxs = NNP.train_idxs.astype(np.int32)

