from abc import ABC, abstractmethod
from PIML.crust.data.nndata.basennprep import StellarNNPrep
from PIML.crust.process.nnprepprocess import NNPrepProcess, StellarNNPrepProcess

from PIML.gateway.loaderIF.nnpreploaderIF import StellarNNPrepLoaderIF
from PIML.gateway.processIF.baseprocessIF import ProcessIF


class NNPrepProcessIF(ProcessIF):
    @abstractmethod
    def interact_on_NNPrep(self, param, data):
        pass

class StellarNNPrepProcessIF(NNPrepProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.loader = StellarNNPrepLoaderIF()   
        self.Process = StellarNNPrepProcess()

    def set_data(self, DATA_PARAMS):
        pass

    def set_param(self, OP_PARAMS):
        pass

    def set_model(self, MODEL_TYPES):
        pass
        # self.OP_MODELS = MODEL_TYPES

    def interact_on_NNPrep(self, PARAMS, NNPrep: StellarNNPrep):
        self.setup(PARAMS)
        self.interact_on_object(NNPrep)

