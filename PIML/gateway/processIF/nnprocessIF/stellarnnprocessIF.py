
from abc import ABC, abstractmethod
from PIML_learn.crust.process.stellarnnprocess import StellarNNProcess

from PIML.gateway.processIF.baseprocessIF import ProcessIF
from PIML.gateway.loaderIF.nnloaderIF.stellarnnloaderIF import StellarNNLoaderIF
# from PIML.gateway.storerIF. import 

from abc import abstractmethod


class NNProcessIF(ProcessIF):
    @abstractmethod
    def interact_on_NN(self, param, NN):
        pass


class StellarNNProcessIF(NNProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.loader = StellarNNLoaderIF()
        self.Process = StellarNNProcess()
        # self.storer = DictStorerIF()

    def set_data(self, DATA_PARAM):
        pass

    def set_param(self, OP_PARAM):
        pass

    def set_model(self, MODEL_TYPES):
        pass
        # self.OP_MODEL = MODEL_TYPES

    def interact_on_NN(self, PARAMS, NN):
        super().setup(PARAMS)
        super().interact_on_Object(NN)
        self.Object = NN

    def finish(self, ext=".h5"):
        pass

