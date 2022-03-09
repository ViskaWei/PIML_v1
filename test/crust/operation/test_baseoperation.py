import numpy as np
from test.testbase import TestBase
from PIML.crust.operation.baseoperation import BaseOperation, SplitOperation

class TestBaseOperation(TestBase):

    def test_BaseOperation(self):
        pass

    def test_SplitOperation(self):
        wave_rng = self.OP_PARAMS["wave_rng"]
        
        OP = SplitOperation(wave_rng)
        wave_new = OP.perform(self.wave)

        self.assertIsNone(np.testing.assert_array_less(wave_rng[0], wave_new))
        self.assertIsNone(np.testing.assert_array_less(wave_new, wave_rng[1]))
        self.assertEqual(OP.split_idxs.shape, (2,))



