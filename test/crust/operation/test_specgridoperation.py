
import numpy as np
from test.testbase import TestBase
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.operation.specgridoperation import BaseSpecGridOperation, SplitSpecGridOperation, TuneSpecGridOperation, BoxSpecGridOperation, CoordxifySpecGridOperation

class TestSpecGridOperation(TestBase):
    



    def test_CoordxifySpecGridOperation(self):
        SpecGrid = self.get_SpecGrid()
        OP = CoordxifySpecGridOperation()
        OP.perform_on_SpecGrid(SpecGrid)
        self.check_Grid(SpecGrid)
        
    def check_Grid(self, SpecGrid: StellarSpecGrid):
        self.assertIsNotNone(SpecGrid.coordx)
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coordx, self.pdx0))
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coordx, SpecGrid.coordx_scaler(SpecGrid.coord)))
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coord, SpecGrid.coordx_rescaler(SpecGrid.coordx)))




    def test_SplitSpecGridOperation(self):
        SpecGrid = self.get_SpecGrid()
        OP = SplitSpecGridOperation(self.OP_PARAMS["arm"])
        OP.perform_on_SpecGrid(SpecGrid)

        self.assertIsNone(np.testing.assert_array_less(OP.rng[0], SpecGrid.wave))
        self.assertIsNone(np.testing.assert_array_less(SpecGrid.wave, OP.rng[1]))
        self.assertEqual(OP.split_idxs.shape, (2,))

            
