import numpy as np
from test.testbase import TestBase
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.process.baseprocess import StellarProcess

class TestBaseProcess(TestBase):
    def test_BaseProcessIF(self):
        pass

    def test_StellarProcess(self):
        self.check_StellarProcess_on_Spec()
        self.check_StellarProcess_on_SpecGrid()

        

    def check_StellarProcess_on_Spec(self):
        spec = StellarSpec(self.wave, self.flux)

        Process = StellarProcess()
        Process.set_process(self.PARAMS, self.MODEL_TYPES)
        Process.start_on_Spec(spec)

        self.assertIsNotNone(spec.wave)
        self.assertIsNotNone(spec.flux)
        # self.assertTrue(spec.wave.shape[0] - self.wave[::self.PARAMS["step"]].shape[0] <=1)
        # self.assertTrue(spec.flux.shape[-1] - self.flux[..., ::self.PARAMS["step"]].shape[-1] <=1)

        

    def check_StellarProcess_on_SpecGrid(self):
        specGrid = StellarSpecGrid(self.wave, self.flux, self.para, self.pdx)

        Process = StellarProcess()
        Process.set_process(self.PARAMS, self.MODEL_TYPES)
        Process.start_on_Spec(specGrid)
        
        self.assertIsNotNone(specGrid.wave)
        self.assertIsNotNone(specGrid.flux)
        self.assertIsNone(np.testing.assert_array_equal(specGrid.coord, self.para))
        self.assertIsNone(np.testing.assert_array_equal(specGrid.coord_idx, self.pdx))