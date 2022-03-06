import numpy as np
from test.testbase import TestBase
from PIML.crust.data.spec.basegrid import StellarGrid
from PIML.gateway.processIF.gridprocessIF.basegridprocessIF import BaseGridProcessIF, GridProcessIF

class test_BaseGridProcessIF(TestBase):


    def test_GridProcessIF(self):
        test_idx = np.array([9,2,2,11,7])
        PARAM = {"IdxInBox": test_idx}
        grid = StellarGrid(self.para, self.pdx)
        GPIF = GridProcessIF()
        GPIF.set_process_param(PARAM)
        GPIF.process_grid(grid)
        coord = grid.coord
        np.testing.assert_array_equal(coord, self.para[test_idx])

