import numpy as np
from test.testbase import TestBase

from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.crust.process.specgridprocess import StellarSpecGridProcess


class TestStellarSpecGridProcess(TestBase):
    def test_StellarSpecGridProcess(self):
        SpecGrid = self.get_SpecGrid()

        SP = StellarSpecGridProcess()
        SP.set_process(self.D.SPECGRID_PARAM, self.D.SPECGRID_MODEL, self.D.SPECGRID_DATA)
        SP.start(SpecGrid)
        self.check_StellarSpecGrid(SpecGrid)



    # def 