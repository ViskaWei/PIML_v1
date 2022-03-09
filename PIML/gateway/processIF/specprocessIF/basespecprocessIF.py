import numpy as np
from abc import abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.process.baseprocess import StellarSpecProcess
from PIML.gateway.processIF.baseprocessIF import BaseProcessIF
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF, SpecGridLoaderIF


class BaseSpecProcessIF(BaseProcessIF):
    """ Base class for process interface for Spec object only. """
    def interact(self, param, data):
        pass

class StellarSpecProcessIF(BaseSpecProcessIF):
    """ class for spectral process. 
        PARAMS = {"arm": wave arm, "step": int }
        MODEL_TYPES = {"Resolution": ResolutionModel (Alex, etc.)}
    
    """
    def __init__(self) -> None:
        super().__init__()
        self.OP_PARAMS: dict = {}
        self.Process = StellarSpecProcess()

    def setup(self, PARAMS, MODEL_TYPES):
        self.set_data(PARAMS["data"])
        self.set_param(PARAMS["op"])
        self.set_model(MODEL_TYPES)

    def set_data(self, DATA_PARAMS):
        self.DATA_PATH = DATA_PARAMS["DATA_PATH"]
        SGL = SpecGridLoaderIF()
        SGL.set_data_path(self.DATA_PATH)
        self.Spec = SGL.load()

    def set_param(self, OP_PARAMS):
        self.OP_PARAMS = self.paramIF(OP_PARAMS)
    
    def set_model(self, MODEL_TYPES):
        self.OP_MODELS = MODEL_TYPES

    def interact(self, PARAMS, MODEL_TYPES):
        self.setup(PARAMS, MODEL_TYPES)
        self.Process.set_process(self.OP_PARAMS, self.OP_MODELS)
        self.Process.start(self.Spec)

    def interact_on_Spec(self, PARAMS, MODEL_TYPES, Spec: StellarSpec):
        self.set_param(PARAMS["op"])
        self.set_model(MODEL_TYPES)
        self.Process.set_process(self.OP_PARAMS, self.OP_MODELS)
        self.Process.start(Spec)

    def paramIF(self, PARAMS):
        #TODO create class later
        
        return PARAMS

