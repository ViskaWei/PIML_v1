import numpy as np
from test.testbase import TestBase

from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.spec.basespecgrid import StellarSpecGrid
from PIML.gateway.processIF.specprocessIF.basespecprocessIF import BaseSpecProcessIF, TrimmableSpecProcessIF, BoxableSpecProcessIF, ResTunableSpecProcessIF

class test_BaseSpecProcessIF(TestBase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.testSpec = StellarSpec(wave=self.wave, flux=self.flux)
        self.testSpecGrid = StellarSpecGrid(wave=self.wave, flux=self.flux, para = self.para)
        self.checkWave = np.copy(self.wave)


    def test_BaseSpecProcessIF(self):
        pass

    def test_ResTunableSpecProcessIF(self):
        RSPIF = ResTunableSpecProcessIF()
        RSPIF.set_spec_process_param({"step":10})
        RSPIF.process_spec(self.spec)

