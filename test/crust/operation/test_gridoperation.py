
import numpy as np
from test.testbase import TestBase
from PIML.crust.data.constants import Constants
from PIML.crust.data.grid.basegrid import BaseGrid, StellarGrid
from PIML.crust.operation.gridoperation import BaseGridOperation, CoordxifyGridOperation

class TestGridOperation(TestBase):
    
    def test_CoordxifyGridOperation(self):
        Grid = self.get_Grid()
        origin = Grid.coord.min(0)
        tick   = Constants.PHYTICK

        OP = CoordxifyGridOperation(origin, tick)
        coordx = OP.perform(Grid.coord)
        self.same_array(coordx, self.D.pdx0)

        OP.perform_on_Grid(Grid)
        self.check_Grid(Grid)
        
    def check_Grid(self, Grid: BaseGrid):
        self.assertIsNotNone(Grid.coordx)
        self.same_array(Grid.coordx, self.D.pdx0)
        self.same_array(Grid.coordx_rng, self.D.pdx0[-1])
        self.same_array(Grid.coordx, Grid.coordx_scaler(Grid.coord))
        self.same_array(Grid.coord, Grid.coordx_rescaler(Grid.coordx))

