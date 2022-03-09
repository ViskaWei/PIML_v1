
import numpy as np
from test.testbase import TestBase
from PIML.crust.operation.specoperation import BaseSpecOperation, SplitSpecOperation, TuneSpecOperation
from PIML.crust.model.spec.resolutionmodel import ResolutionModel


class TestSpecOperation(TestBase):

    def test_SplitSpecOperation(self):
        
        OP = SplitSpecOperation(self.arm)
        
        wave_new = OP.perform(self.wave)
        self.assertIsNotNone(wave_new)

        Spec = self.check_perform_on_spec(OP)
        self.assertIsNone(np.testing.assert_array_less(OP.rng[0], Spec.wave))
        self.assertIsNone(np.testing.assert_array_less(Spec.wave, OP.rng[1]))
        self.assertEqual(OP.split_idxs.shape, (2,))
        
    def test_TuneSpecOperation(self):
        model_type, model_param = "Alex", 10
        
        OP = TuneSpecOperation(model_type, model_param)
        wave_new = OP.perform(self.wave)
        
        self.assertIsInstance(OP.model, ResolutionModel)
        self.assertIsNotNone(wave_new)


        Spec = self.check_perform_on_spec(OP)
        self.assertTrue(Spec.wave.shape[0] - self.wave[::model_param].shape[0] <=1)


    def check_perform_on_spec(self, OP):
        Spec = self.get_Spec()
        OP.perform_on_Spec(Spec)
        
        self.assertIsNotNone(Spec.wave)
        self.assertIsNotNone(Spec.flux)
        return Spec
