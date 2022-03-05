import numpy as np
from test.testbase import TestBase

from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.spec.basespecgrid import StellarSpecGrid
from PIML.gateway.modelIF.specmodelIF.basespecmodelIF import ResolutionSpecModelIF


class TestBaseSpecModel(TestBase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.testSpec = StellarSpec(wave=self.wave, flux=self.flux)
        self.testSpecGrid = StellarSpecGrid(wave=self.wave, flux=self.flux, para = self.para)
        self.checkWave = np.copy(self.wave)

    def test_ResolutionSpecModelIF(self):
        step = 10
        resSpecModelIF = ResolutionSpecModelIF()
        resSpecModelIF.set_spec_model("Alex")
        resSpecModelIF.set_spec_model_param({"step": step})
        #==========================================================
        spec = resSpecModelIF.apply_model_on_spec(self.testSpec)

        self.assertTrue(spec.wave.shape[0] - self.specGrid.wave[::step].shape[0] <=1)
        self.assertTrue(spec.flux.shape[0] == self.specGrid.flux[..., ::step].shape[0])
        self.assertTrue(spec.flux.shape[1] - self.specGrid.flux[..., ::step].shape[1] <=1)
        np.testing.assert_array_equal(self.wave, self.checkWave)

        #==========================================================
        specGrid = resSpecModelIF.apply_model_on_spec(self.testSpecGrid)

        self.assertTrue(specGrid.wave.shape[0] - self.specGrid.wave[::step].shape[0] <=1)
        self.assertTrue(specGrid.flux.shape[0] == self.specGrid.flux[..., ::step].shape[0])
        self.assertTrue(specGrid.flux.shape[1] - self.specGrid.flux[..., ::step].shape[1] <=1)
        np.testing.assert_array_equal(self.wave, self.checkWave)


        #==========================================================
        resSpecModelIF.set_spec_model_data(self.specGrid)
        wave, flux = resSpecModelIF.apply_spec_model(resSpecModelIF.wave,\
                                                    resSpecModelIF.flux)

        self.assertTrue(wave.shape[0] - self.specGrid.wave[::step].shape[0] <=1)
        self.assertTrue(flux.shape[0] == self.specGrid.flux[..., ::step].shape[0])
        self.assertTrue(flux.shape[1] - self.specGrid.flux[..., ::step].shape[1] <=1)

        #==========================================================
        resSpecModelIF.set_spec_model_data(self.spec)
        wave, flux = resSpecModelIF.apply_spec_model(resSpecModelIF.wave,\
                                                    resSpecModelIF.flux)
        self.assertTrue(wave.shape[0] - self.spec.wave[::step].shape[0] <=1)
        self.assertTrue(flux.shape[0] == self.spec.flux[..., ::step].shape[0])
        self.assertTrue(flux.shape[1] - self.spec.flux[..., ::step].shape[1] <=1)

        np.testing.assert_array_equal(self.wave, self.checkWave)


    

