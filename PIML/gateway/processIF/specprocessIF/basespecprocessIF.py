import numpy as np

from abc import abstractmethod

from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.spec.basegrid import StellarGrid
from PIML.crust.process.baseprocess import StellarProcess

from PIML.gateway.processIF.baseprocessIF import BaseProcessIF

class BaseSpecProcessIF(BaseProcessIF):
    """ Base class for process interface for Spec object only. """
    def interact(self, param, data):
        pass

class StellarProcessIF(BaseSpecProcessIF):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.OP_PARAMS: dict = {}
        self.Process = StellarProcess()

    def interact(self, PARAMS, MODEL_TYPES, Spec: StellarSpec):
        self.OP_PARAMS = self.paramIF(PARAMS, Spec)
        self.OP_MODEL = MODEL_TYPES
        self.Process.set_process(self.OP_PARAMS, MODEL_TYPES)
        self.Process.start_on_Spec(Spec)

    def paramIF(self, PARAMS, Spec):
        #TODO create class later
        wRng = PARAMS["wave_rng"]
        split_idxs = np.digitize(Spec.wave, wRng)
        step = PARAMS["step"]
        return {"split_idxs": split_idxs, "step": step}

