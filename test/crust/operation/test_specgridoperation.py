
import numpy as np
from test.testbase import TestBase
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.crust.operation.specgridoperation import BaseSpecGridOperation, SplitSpecGridOperation, TuneSpecGridOperation, BoxSpecGridOperation, CoordxifySpecGridOperation

class TestSpecGridOperation(TestBase):
    
    def test_CoordxifySpecGridOperation(self):
        SpecGrid = self.get_SpecGrid()
        OP = CoordxifySpecGridOperation()
        OP.perform_on_SpecGrid(SpecGrid)
        self.check_Grid(SpecGrid)
        
    def check_Grid(self, SpecGrid: StellarSpecGrid):
        self.assertIsNotNone(SpecGrid.coordx)
        self.assertTrue((SpecGrid.coordx[0] == 0).all())
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coordx, self.D.pdx0))
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coordx, SpecGrid.coordx_scaler(SpecGrid.coord)))
        self.assertIsNone(np.testing.assert_array_equal(SpecGrid.coord, SpecGrid.coordx_rescaler(SpecGrid.coordx)))

    def test_SplitSpecGridOperation(self):
        SpecGrid = self.get_SpecGrid()
        OP = SplitSpecGridOperation(self.D.SPECGRID_PARAM["arm"])
        OP.perform_on_SpecGrid(SpecGrid)

        self.assertTrue(OP.rng[0], SpecGrid.wave)
        self.assertIsNone(np.testing.assert_array_less(SpecGrid.wave, OP.rng[1]))
        self.assertEqual(OP.split_idxs.shape, (2,))

            
