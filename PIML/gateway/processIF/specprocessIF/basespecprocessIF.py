import numpy as np
from abc import abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.process.baseprocess import StellarSpecProcess
from PIML.gateway.processIF.baseprocessIF import BaseProcessIF
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF, SpecLoaderIF, SkyLoaderIF


class BaseSpecProcessIF(BaseProcessIF):
    """ Base class for process interface for Spec object only. """
    def interact_on_Spec(self, param, Spec: StellarSpec):
        pass

class StellarSpecProcessIF(BaseSpecProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.OP_PARAMS: dict = {}
        self.loader = SpecLoaderIF()
        self.Process = StellarSpecProcess()

    def setup(self, PARAMS):
        self.set_data(PARAMS["data"])
        self.set_param(PARAMS["op"])
        self.set_model(PARAMS["model"])

    def set_object(self, OBJECT_PARAMS):
        self.DATA_PATH = OBJECT_PARAMS["DATA_PATH"]
        self.loader.set_data_path(self.DATA_PATH)
        self.Object = self.loader.load()

    def set_data(self, DATA_PARAMS):
        self.SKY_PATH = DATA_PARAMS["SKY_PATH"]
        self.Sky = SkyLoaderIF().load(self.SKY_PATH)
        self.OP_DATA = {"Sky": self.Sky}

    def set_param(self, OP_PARAMS):
        self.OP_PARAMS = self.paramIF(OP_PARAMS)
    
    def set_model(self, MODEL_TYPES):
        self.OP_MODELS = MODEL_TYPES

    def interact(self, PARAMS):
        self.set_object(PARAMS["object"])
        self.setup(PARAMS)
        self.Process.set_process(self.OP_PARAMS, self.OP_MODELS, self.OP_DATA)
        self.Process.start(self.Object)

    def interact_on_Spec(self, PARAMS, Spec: StellarSpec):
        self.setup(PARAMS)
        self.Process.set_process(self.OP_PARAMS, self.OP_MODELS, self.OP_DATA)
        self.Process.start(Spec)

    def paramIF(self, PARAMS):
        #TODO create class later
        
        return PARAMS

