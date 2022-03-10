from test.testbase import TestBase
from PIML.crust.model.basemodel import BaseModel
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
# from PIML.crust.model.specgrid.interpspecgridmodel import InterpModel, RBFInterpModel, PCARBFInterpModel
from PIML.crust.model.specgrid.basespecgridmodel import InterpSpecGridModel

class TestBaseSpecGridModel(TestBase):
    
    def test_BaseSpecGridModel(self):
        pass

    def test_InterpSpecGridmodel(self):
        # SpecGrid = StellarSpecGrid(self.wave, self.flux[:100], self.para[:100], self.pdx[:100])
        # model = InterpSpecGridModel("RBF", None)
        # model.apply_on_SpecGrid(SpecGrid)
        # self.assertIsNotNone(SpecGrid.interpolator)
        pass




