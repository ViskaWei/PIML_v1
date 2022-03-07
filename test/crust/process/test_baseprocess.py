import numpy as np
from test.testbase import TestBase
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.spec.basespecgrid import StellarSpecGrid
from PIML.crust.process.baseprocess import StellarProcess

class BaseProcessTest(TestBase):
    def test_BaseProcessIF(self):
        pass

    def test_StellarProcess(self):
        Process = StellarProcess()
        Process.set_process(self.PARAMS, self.MODEL_TYPES)
        flux_test = np.copy(self.flux)
        flux_new = Process.start(flux_test)
        self.assertIsNone(np.testing.assert_array_equal(flux_new, self.flux))

        spec = StellarSpec(self.wave, self.flux)
        Process.start_on_Spec(spec)
        self.assertIsNotNone(spec.wave)
        self.assertIsNotNone(spec.flux)
        self.assertTrue(spec.wave.shape[0] - self.wave[::self.PARAMS["step"]].shape[0] <=1)
        self.assertTrue(spec.flux.shape[-1] - self.flux[..., ::self.PARAMS["step"]].shape[-1] <=1)


        specGrid = StellarSpecGrid(self.wave, self.flux, self.para, self.pdx)
        Process.start_on_Spec(specGrid)
        self.assertIsNotNone(specGrid.wave)
        self.assertIsNotNone(specGrid.flux)

        self.assertTrue(spec.wave.shape[0] - self.wave[::self.PARAMS["step"]].shape[0] <=1)
        self.assertTrue(spec.flux.shape[-1] - self.flux[..., ::self.PARAMS["step"]].shape[-1] <=1)


        self.assertIsNone(np.testing.assert_array_equal(specGrid.para, self.para))
        self.assertIsNone(np.testing.assert_array_equal(specGrid.pdx, self.pdx))