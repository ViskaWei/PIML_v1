import numpy as np
from test.testbase import TestBase
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.operation.baseoperation import BaseOperation, BaseSpecOperation, SplitOperation, ResolutionOperation
from PIML.crust.model.spec.resolutionmodel import ResolutionModel

class TestBaseOperation(TestBase):

    def test_BaseOperation(self):
        pass

    def test_ResolutionOperation(self):
        model_type, model_param = "Alex", 10
        
        OP = ResolutionOperation(model_type, model_param)
        wave_new = OP.perform(self.wave)
        
        self.assertIsInstance(OP.model, ResolutionModel)
        self.assertIsNotNone(wave_new)

        Spec = StellarSpec(self.wave, self.flux)
        OP.perform_on_Spec(Spec)
        
        self.assertIsNotNone(Spec.wave)
        self.assertIsNotNone(Spec.flux)
        self.assertTrue(Spec.wave.shape[0] - self.wave[::model_param].shape[0] <=1)


    def test_SplitOperation(self):
        startIdx, endIdx = self.PARAMS["StartEndIdx"]
        
        OP = SplitOperation(startIdx, endIdx)
        wave_new = OP.perform(self.wave)

        self.assertIsNone(np.testing.assert_array_equal(wave_new, self.wave[startIdx:endIdx]))



