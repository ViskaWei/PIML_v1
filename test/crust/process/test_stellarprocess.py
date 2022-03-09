import numpy as np

from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.process.stellarprocess import StellarProcess
from test.testbase import TestBase


class TestStellarProcess(TestBase):
    def test_StellarProcess(self):
        SP = StellarProcess()
        SP.set_process(self.OP_PARAMS, self.MODEL_TYPES)


    def check_StellarProcess_on_SpecGrid(self):
        specGrid = StellarSpecGrid(self.wave, self.flux, self.para, self.pdx)

        Process = StellarProcess()
        Process.set_process(self.OP_PARAMS, self.MODEL_TYPES)
        Process.start(specGrid)
        
        self.assertIsNone(np.testing.assert_array_less(specGrid.wave.shape, self.wave.shape))
        self.assertIsNone(np.testing.assert_array_less(specGrid.flux.shape[1], self.flux.shape[1]))

        self.assertIsNone(np.testing.assert_array_equal(specGrid.coord, self.para))
        self.assertIsNone(np.testing.assert_array_equal(specGrid.coord_idx, self.pdx))

        self.assertIsNotNone(specGrid.box)

