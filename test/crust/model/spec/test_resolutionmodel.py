
import numpy as np
from test.testbase import TestBase
from PIML.crust.model.spec.resolutionmodel import  AlexResolutionModel, NpResolutionModel

class TestResolutionModel(TestBase):

    def test_AlexResolutionModel(self):
        step = 10

        model = AlexResolutionModel()
        model.set_step(step)

        data = self.wave
        dataNew = model.apply(data)
        dataToCheck = np.diff(np.cumsum(data, axis=-1)[..., ::step], axis=-1) / step
        np.testing.assert_array_equal(dataNew, dataToCheck)

        data = self.flux
        dataNew = model.apply(data)
        dataToCheck = np.diff(np.cumsum(data, axis=-1)[..., ::step], axis=-1) / step
        np.testing.assert_array_equal(dataNew, dataToCheck)

    def test_NpResolutionModel(self):
        pass