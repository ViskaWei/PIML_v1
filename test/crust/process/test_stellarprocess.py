import numpy as np
from test.testbase import TestBase

from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.process.stellarprocess import StellarProcess


class TestStellarProcess(TestBase):
    def test_StellarProcess(self):
        SP = StellarProcess()
        SP.set_process(self.OP_PARAMS, self.MODEL_TYPES)
        self.check_StellarProcess_on_SpecGrid(SP)

    def check_StellarProcess_on_SpecGrid(self, Process: StellarProcess):
        SpecGrid = self.get_SpecGrid()
        Process.start(SpecGrid)
        
        self.assertIsNone(np.testing.assert_array_less(SpecGrid.wave.shape, self.wave.shape))
        self.assertIsNone(np.testing.assert_array_less(SpecGrid.flux.shape[1], self.flux.shape[1]))

        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coord    , self.para))
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coord_idx, self.pdx))

        self.assertIsNotNone(SpecGrid.box)
        self.assertIsNotNone(SpecGrid.coordx)
        self.assertTrue((SpecGrid.coordx[0] == 0).all())
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coordx, self.pdx0))
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coordx, SpecGrid.coordx_scaler(SpecGrid.coord)))
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coord, SpecGrid.coordx_rescaler(SpecGrid.coordx)))


    # def 