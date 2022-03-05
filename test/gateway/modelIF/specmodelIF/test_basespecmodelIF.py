import numpy as np
from PIML.gateway.modelIF.specmodelIF.basespecmodelIF import ResolutionSpecModelIF
from test.testbase import TestBase


class TestBaseSpecModel(TestBase):

    def test_ResolutionSpecModelIF(self):
        step = 10
        resSpecModelIF = ResolutionSpecModelIF()
        resSpecModelIF.set_model("Alex")
        resSpecModelIF.set_model_param({"step": step})
        resSpecModelIF.set_model_data(self.specGrid)
        wave, flux = resSpecModelIF.apply_model()
        assert wave.shape == self.specGrid.wave[::step].shape
        assert flux.shape == self.specGrid.flux[..., ::step].shape

        resSpecModelIF.set_model_data(self.spec)
        wave, flux = resSpecModelIF.apply_model()
        assert wave.shape == self.spec.wave[::step].shape
        assert flux.shape == self.spec.flux[..., ::step].shape




