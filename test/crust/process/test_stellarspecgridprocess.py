import numpy as np
from test.testbase import TestBase

from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.crust.process.stellarspecgridprocess import StellarSpecGridProcess


class TestStellarSpecGridProcess(TestBase):
    def test_StellarSpecGridProcess(self):
        SP = StellarSpecGridProcess()
        SP.set_process(self.D.OP_PARAMS, self.D.OP_MODELS, self.D.OP_DATA)
        self.check_StellarSpecGridProcess_on_SpecGrid(SP)

    def check_StellarSpecGridProcess_on_SpecGrid(self, Process: StellarSpecGridProcess):
        SpecGrid = self.get_SpecGrid()
        Process.start(SpecGrid)
        
        self.assertEqual(SpecGrid.wave.shape, self.D.wave_RedM.shape)
        self.assertEqual(SpecGrid.flux.shape[1], self.D.wave_RedM.shape[0])
        self.same_array(SpecGrid.coord    , self.D.para)
        self.same_array(SpecGrid.coord_idx, self.D.pdx)
        self.same_array(SpecGrid.coordx, self.D.pdx0)
        self.same_array(SpecGrid.coordx, SpecGrid.coordx_scaler(SpecGrid.coord))
        self.same_array(SpecGrid.coord, SpecGrid.coordx_rescaler(SpecGrid.coordx))

        self.assertIsNotNone(SpecGrid.box)
        self.assertIsNotNone(SpecGrid.logflux)
        self.assertIsNotNone(SpecGrid.Obs)
        

    # def 