import numpy as np
from test.testbase import TestBase

from PIML.crust.data.specdata.basespec import StellarSpec
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.gateway.processIF.specprocessIF.basespecprocessIF import BaseSpecProcessIF, StellarSpecProcessIF

class TestBaseSpecProcessIF(TestBase):
        
    def test_BaseSpecProcessIF(self):
        pass

    def test_StellarSpecProcessIF(self):
        testSpec = self.get_Spec()
        wave_to_check = np.copy(testSpec.wave)

        PIF = StellarSpecProcessIF()
        PIF.interact_on_Spec(self.D.PARAMS, testSpec)
        # self.assertIsNone(np.testing.assert_array_equal(checkWave, self.D.wave))
        self.assertIsNotNone(testSpec.wave)
        self.assertIsNotNone(testSpec.flux)

        # PIF = StellarSpecProcessIF()
        # PIF.interact_on_Spec(self.PARAMS, self.OP_MODELS, self.testSpecGrid)
        # self.assertIsNone(np.testing.assert_array_equal(self.testSpecGrid.coord, self.para))
        # self.assertIsNone(np.testing.assert_array_equal(self.testSpecGrid.coord_idx, self.pdx))

        # self.assertIsNotNone(self.testSpecGrid.wave)
        # self.assertIsNotNone(self.testSpecGrid.flux)


    
