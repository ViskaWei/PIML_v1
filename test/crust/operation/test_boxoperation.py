
import numpy as np
from test.testbase import TestBase
from PIML.crust.operation.boxoperation import BaseBoxOperation, StellarBoxOperation

class TestBoxOperation(TestBase):

    def test_StellarBoxOperation(self):
        Grid = self.get_Grid()
        para = np.copy(Grid.coord)
        box_name = "R"
        OP = StellarBoxOperation(box_name)

        box_dict = OP.perform(Grid.dfcoord)
        self.check_box_output(box_dict, para)

        OP.perform_on_Box(Grid)
        self.check_box_output(Grid.box, para)

    def check_box_output(self, box_dict, para):
        self.assertIsNotNone(box_dict)
        self.assertIsNotNone(box_dict["name"])

        self.same_array(box_dict["min"], para.min(axis=0))
        self.same_array(box_dict["max"], para.max(axis=0))

        self.assertIsNotNone(box_dict["rng"])
        self.assertIsNotNone(box_dict["num"])
        self.assertIsNotNone(box_dict["mid"])


