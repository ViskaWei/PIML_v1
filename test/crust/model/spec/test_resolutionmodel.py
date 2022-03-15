
import numpy as np
from test.testbase import TestBase
from PIML.crust.model.spec.resolutionmodel import  AlexResolutionModel, NpResolutionModel

class TestResolutionModel(TestBase):

    def test_AlexResolutionModel(self):
        step = 10

        model = AlexResolutionModel()
        assert (model.name == "Alex")

        data = self.wave
        dataNew = model.tune_resolution(data, step)
        dataToCheck = np.diff(np.cumsum(data, axis=-1)[..., ::step], axis=-1) / step
        self.assertIsNone(np.testing.assert_array_equal(dataNew, dataToCheck))

        data = self.flux
        dataNew = model.tune_resolution(data, step)        
        dataToCheck = np.diff(np.cumsum(data, axis=-1)[..., ::step], axis=-1) / step
        self.assertIsNone(np.testing.assert_array_equal(dataNew, dataToCheck))


    def test_NpResolutionModel(self):
        pass