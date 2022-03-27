from abc import ABC, abstractmethod
from PIML.crust.data.nndata.basennprep import NNPrep
from PIML.crust.process.nnprepprocess import StellarNNPrepProcess

from PIML.gateway.loaderIF.nnpreploaderIF import StellarNNPrepLoaderIF
from PIML.gateway.processIF.baseprocessIF import ProcessIF
from PIML.gateway.storerIF.basestorerIF import PickleStorerIF


class NNPrepProcessIF(ProcessIF):
    @abstractmethod
    def interact_on_NNPrep(self, param, data):
        pass

class StellarNNPrepProcessIF(NNPrepProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.loader  = StellarNNPrepLoaderIF()   
        self.Process = StellarNNPrepProcess()
        self.storer  = PickleStorerIF()

    def set_data(self, DATA_PARAM):
        self.OP_DATA["rng"]  = DATA_PARAM["rng"]

    def set_param(self, OP_PARAM):
        self.OP_PARAM["step"] = OP_PARAM["step"]
        pass

    def set_model(self, MODEL_TYPES):
        pass
        # self.OP_MODEL = MODEL_TYPES

    def interact_on_NNPrep(self, PARAMS, NNPrep: NNPrep):
        super().setup(PARAMS)
        super().interact_on_Object(NNPrep)
        self.Object = NNPrep
        

    def finish(self, store_path, name="NNPrep"):
        self.storer.set_path(store_path, name)
        self.storer.store(self.Object)
        