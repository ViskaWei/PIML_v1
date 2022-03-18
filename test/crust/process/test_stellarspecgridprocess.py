import numpy as np
from test.testbase import TestBase

from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.crust.process.stellarspecgridprocess import StellarSpecGridProcess


class TestStellarSpecGridProcess(TestBase):
    def test_StellarSpecGridProcess(self):
        SpecGrid = self.get_SpecGrid()

        SP = StellarSpecGridProcess()
        SP.set_process(self.D.OP_PARAMS, self.D.OP_MODELS, self.D.OP_DATA)
        SP.start(SpecGrid)
        self.check_StellarSpecGrid(SpecGrid)



    # def 