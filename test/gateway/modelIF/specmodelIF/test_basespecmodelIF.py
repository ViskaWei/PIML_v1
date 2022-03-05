import numpy as np
from PIML.gateway.modelIF.specmodelIF.basespecmodelIF import ResolutionSpecModelIF
from test.testbase import TestBase


class TestBaseSpecModel(TestBase):

    def test_ResolutionSpecModelIF(self):
        step = 10
        resSpecModelIF = ResolutionSpecModelIF()
        resSpecModelIF.set_spec_model("Alex")
        resSpecModelIF.set_spec_model_param({"step": step})
        resSpecModelIF.set_spec_model_data(self.specGrid)
        wave, flux = resSpecModelIF.apply_spec_model()

        self.assertTrue(wave.shape[0] - self.specGrid.wave[::step].shape[0] <=1)
        self.assertTrue(flux.shape[0] == self.specGrid.flux[..., ::step].shape[0])
        self.assertTrue(flux.shape[1] - self.specGrid.flux[..., ::step].shape[1] <=1)

        resSpecModelIF.set_spec_model_data(self.spec)
        wave, flux = resSpecModelIF.apply_spec_model()
        self.assertTrue(wave.shape[0] - self.spec.wave[::step].shape[0] <=1)
        self.assertTrue(flux.shape[0] == self.spec.flux[..., ::step].shape[0])
        self.assertTrue(flux.shape[1] - self.spec.flux[..., ::step].shape[1] <=1)



    

