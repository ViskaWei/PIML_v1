import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.process.stellarprocess import StellarProcess

from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF, SpecGridLoaderIF
from PIML.gateway.processIF.baseprocessIF import BaseProcessIF


class BaseSpecGridProcessIF(BaseProcessIF):
    pass

class StellarProcessIF(BaseSpecGridProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.OP_PARAMS: dict = {}
        self.Process = StellarProcess()

    def setup(self, PARAMS, MODEL_TYPES):
        self.set_data(PARAMS["data"])
        self.set_param(PARAMS["op"])
        self.set_model(MODEL_TYPES)

    def set_data(self, DATA_PARAMS):
        self.DATA_PATH = DATA_PARAMS["DATA_PATH"]
        SGL = SpecGridLoaderIF()
        SGL.set_data_path(self.DATA_PATH)
        self.SpecGrid = SGL.load()

    def set_param(self, OP_PARAMS):
        self.OP_PARAMS = self.paramIF(OP_PARAMS)
    
    def set_model(self, MODEL_TYPES):
        self.OP_MODELS = MODEL_TYPES

    def interact(self, PARAMS, MODEL_TYPES):
        self.setup(PARAMS, MODEL_TYPES)
        self.Process.set_process(self.OP_PARAMS, self.OP_MODELS)
        self.Process.start(self.SpecGrid)

    def interact_on_Spec(self, PARAMS, MODEL_TYPES, SpecGrid: StellarSpecGrid):
        self.set_param(PARAMS["op"])
        self.set_model(MODEL_TYPES)
        self.Process.set_process(self.OP_PARAMS, self.OP_MODELS)
        self.Process.start(SpecGrid)

    def paramIF(self, PARAMS):
        #TODO create class later
        
        return PARAMS

