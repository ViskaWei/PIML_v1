import numpy as np
from test.testbase import TestBase

from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.process.stellarprocess import StellarProcess


class TestStellarProcess(TestBase):
    def test_StellarProcess(self):
        SP = StellarProcess()
        SP.set_process(self.OP_PARAMS, self.OP_MODELS, self.OP_DATA)
        self.check_StellarProcess_on_SpecGrid(SP)

    def check_StellarProcess_on_SpecGrid(self, Process: StellarProcess):
        SpecGrid = self.get_SpecGrid()
        Process.start(SpecGrid)
        
        self.same_array(SpecGrid.wave.shape, self.wave.shape)
        self.same_array(SpecGrid.flux.shape[1], self.flux.shape[1])
        self.same_array(SpecGrid.coord    , self.para)
        self.same_array(SpecGrid.coord_idx, self.pdx)

        self.assertIsNotNone(SpecGrid.box)
        self.assertIsNotNone(SpecGrid.coordx)
        self.assertTrue((SpecGrid.coordx[0] == 0).all())

        self.same_array(SpecGrid.coordx, self.pdx0)
        self.same_array(SpecGrid.coordx, SpecGrid.coordx_scaler(SpecGrid.coord))
        self.same_array(SpecGrid.coord, SpecGrid.coordx_rescaler(SpecGrid.coordx))


    # def 