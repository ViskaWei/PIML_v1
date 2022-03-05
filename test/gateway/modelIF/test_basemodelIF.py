import numpy as np
from test.testbase import TestBase
from PIML.gateway.modelIF.basemodelIF import BaseModelIF, ResolutionModelIF

class TestBaseModel(TestBase):
    
    def test_ResolutionModelIF(self):
        resModelIF = ResolutionModelIF()

        self.assertRaises(ValueError, resModelIF.set_model, "test")
        resModelIF.set_model("Alex")
        
        step = 10
        resModelIF.set_model_param({"step": step})

        wave = resModelIF.apply_model(self.wave)
        self.assertTrue(wave.shape[0] - self.wave[::step].shape[0] <=1)

        flux = resModelIF.apply_model(self.flux)
        self.assertTrue(flux.shape[0] == self.flux[..., ::step].shape[0])
        self.assertTrue(flux.shape[1] - self.flux[..., ::step].shape[1] <=1)


    def test_BaseModelIF(self):
        for ModelIF in BaseModelIF.__subclasses__():
            modelIF = ModelIF()
            pass

