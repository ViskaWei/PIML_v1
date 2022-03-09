
import numpy as np
from test.testbase import TestBase
from PIML.crust.data.constants import Constants

from PIML.crust.data.grid.basegrid import BaseGrid, StellarGrid

from PIML.crust.operation.boxoperation import BaseBoxOperation, StellarBoxOperation

class TestBoxOperation(TestBase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.Grid = StellarGrid(self.para, self.pdx)

    def test_StellarBoxOperation(self):
        param = self.OP_PARAMS["box_name"]
        OP = StellarBoxOperation(param)

        box_dict = OP.perform(self.Grid.dfcoord)
        self.check_box_output(box_dict)

        OP.perform_on_Box(self.Grid)
        self.check_box_output(self.Grid.box)

    def check_box_output(self, box_dict):
        self.assertIsNotNone(box_dict)
        self.assertIsNotNone(box_dict["name"])
        self.assertIsNone(np.testing.assert_array_equal(box_dict["min"], self.para.min(axis=0)))
        self.assertIsNone(np.testing.assert_array_equal(box_dict["max"], self.para.max(axis=0)))
        self.assertIsNotNone(box_dict["rng"])
        self.assertIsNotNone(box_dict["num"])
        self.assertIsNotNone(box_dict["mid"])


