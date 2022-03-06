import numpy as np
from test.testbase import TestBase

from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.spec.basespecgrid import StellarSpecGrid
from PIML.gateway.processIF.specprocessIF.basespecprocessIF import BaseSpecProcessIF, TrimmableSpecProcessIF, BoxableSpecProcessIF, ResTunableSpecProcessIF

class test_BaseSpecProcessIF(TestBase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.testSpec = StellarSpec(self.wave, self.flux)
        self.testSpecGrid = StellarSpecGrid(self.wave, self.flux, self.para, self.pdx)
        self.checkWave = np.copy(self.wave)


    def test_BaseSpecProcessIF(self):
        pass

    def test_ResTunableSpecProcessIF(self):
        MODEL_TYPES = {
            "ResTunableProcess": "Alex"
        }
        PARAMS = {"step": 10}
        RSPIF = ResTunableSpecProcessIF()
        RSPIF.set_process_model(MODEL_TYPES)
        RSPIF.set_process_param(PARAMS)
        RSPIF.process_spec(self.spec)


    def test_