import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.process.stellarprocess import StellarProcess
from PIML.gateway.loaderIF.baseloaderIF import SpecGridLoaderIF
from PIML.gateway.processIF.baseprocessIF import BaseProcessIF
from PIML.gateway.processIF.specprocessIF.basespecprocessIF import StellarSpecProcessIF

class BaseSpecGridProcessIF(BaseProcessIF):
    """ Base class for process interface for Spec object only. """
    @abstractmethod
    def interact_on_SpecGrid(self, param, SpecGrid: StellarSpecGrid):
        pass


class StellarProcessIF(StellarSpecProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.OP_PARAMS: dict = {}
        self.loader = SpecGridLoaderIF()
        self.Process = StellarProcess()

    def interact_on_SpecGrid(self, PARAMS, MODEL_TYPES, SpecGrid: StellarSpecGrid):
        self.setup(PARAMS, MODEL_TYPES)
        self.Process.set_process(self.OP_PARAMS, self.OP_MODELS, self.OP_DATA)
        self.Process.start(SpecGrid)


