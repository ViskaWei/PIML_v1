import numpy as np
from test.testbase import TestBase

from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.spec.basespecgrid import StellarSpecGrid
from PIML.gateway.processIF.specprocessIF.basespecprocessIF import BaseSpecProcessIF, StellarProcessIF

class TestBaseSpecProcessIF(TestBase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.testSpec = StellarSpec(self.wave, self.flux)
        self.testSpecGrid = StellarSpecGrid(self.wave, self.flux, self.para, self.pdx)
        self.checkWave = np.copy(self.wave)


    def test_BaseSpecProcessIF(self):
        pass

    def test_StellarProcessIF(self):
        
        PIF = StellarProcessIF()
        PIF.interact(self.PARAMS, self.MODEL_TYPES, self.testSpec)
        self.assertIsNone(np.testing.assert_array_equal(self.checkWave, self.wave))
        self.assertIsNotNone(self.testSpec.wave)
        self.assertIsNotNone(self.testSpec.flux)

        PIF = StellarProcessIF()
        PIF.interact(self.PARAMS, self.MODEL_TYPES, self.testSpecGrid)
        self.assertIsNone(np.testing.assert_array_equal(self.testSpecGrid.coord, self.para))
        self.assertIsNone(np.testing.assert_array_equal(self.testSpecGrid.coord_idx, self.pdx))

        self.assertIsNotNone(self.testSpecGrid.wave)
        self.assertIsNotNone(self.testSpecGrid.flux)


    
