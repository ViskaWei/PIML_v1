
import numpy as np
from test.testbase import TestBase
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.operation.specoperation import BaseSpecOperation, ArmSplitOperation, ResolutionOperation
from PIML.crust.model.spec.resolutionmodel import ResolutionModel


class TestSpecOperation(TestBase):

    def test_ArmSplitOperation(self):
        arm = "arm_test"
        OP = ArmSplitOperation(arm)
        
        wave_new = OP.perform(self.wave)
        self.assertIsNotNone(wave_new)

        Spec = self.check_perform_on_spec(OP)
        self.assertIsNone(np.testing.assert_array_less(OP.rng[0], Spec.wave))
        self.assertIsNone(np.testing.assert_array_less(Spec.wave, OP.rng[1]))




    def test_ResolutionOperation(self):
        model_type, model_param = "Alex", 10
        
        OP = ResolutionOperation(model_type, model_param)
        wave_new = OP.perform(self.wave)
        
        self.assertIsInstance(OP.model, ResolutionModel)
        self.assertIsNotNone(wave_new)


        Spec = self.check_perform_on_spec(OP)
        self.assertTrue(Spec.wave.shape[0] - self.wave[::model_param].shape[0] <=1)


    def check_perform_on_spec(self, OP):
        Spec = StellarSpec(self.wave, self.flux)
        OP.perform_on_Spec(Spec)
        
        self.assertIsNotNone(Spec.wave)
        self.assertIsNotNone(Spec.flux)
        return Spec
