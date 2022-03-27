from abc import ABC, abstractmethod
from PIML.crust.data.nndata.baseprepnn import PrepNN
from PIML.crust.process.prepnnprocess import StellarPrepNNProcess

from PIML.gateway.loaderIF.prepnnloaderIF import StellarPrepNNLoaderIF
from PIML.gateway.processIF.baseprocessIF import ProcessIF
from PIML.gateway.storerIF.basestorerIF import PickleStorerIF


class PrepNNProcessIF(ProcessIF):
    @abstractmethod
    def interact_on_PrepNN(self, param, data):
        pass

class StellarPrepNNProcessIF(PrepNNProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.loader  = StellarPrepNNLoaderIF()   
        self.Process = StellarPrepNNProcess()
        self.storer  = PickleStorerIF()

    

    def set_data(self, DATA_PARAM):
        self.OP_DATA["rng"]     = DATA_PARAM["rng"]

    def set_param(self, OP_PARAM):
        self.OP_PARAM["step"]   = OP_PARAM["step"]
        self.OP_PARAM["ntrain"] = OP_PARAM["ntrain"]
        self.OP_PARAM["ntest"]  = OP_PARAM["ntest"]

    def set_model(self, MODEL_TYPES):
        pass
        # self.OP_MODEL = MODEL_TYPES

    def interact_on_PrepNN(self, PARAMS, PrepNN: PrepNN):
        super().setup(PARAMS)
        super().interact_on_Object(PrepNN)
        self.Object = PrepNN

    def finish(self, store_path, name="PrepNN"):
        self.storer.set_path(store_path, name)
        self.storer.store(self.Object)
        