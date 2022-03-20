
import numpy as np
from test.testbase import TestBase
from PIML.core.method.resolution import  AlexResolution, NpResolution

class TestResolution(TestBase):

    def test_AlexResolution(self):
        step = 10

        model = AlexResolution()
        assert (model.name == "Alex")

        data = self.D.wave
        dataNew = model.tune(data, step)
        dataToCheck = np.diff(np.cumsum(data, axis=-1)[..., ::step], axis=-1) / step
        self.assertIsNone(np.testing.assert_array_equal(dataNew, dataToCheck))

        data = self.D.flux
        dataNew = model.tune(data, step)        
        dataToCheck = np.diff(np.cumsum(data, axis=-1)[..., ::step], axis=-1) / step
        self.assertIsNone(np.testing.assert_array_equal(dataNew, dataToCheck))


    def test_NpResolutionModel(self):
        pass