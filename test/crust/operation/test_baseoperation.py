import numpy as np
from test.testbase import TestBase
from PIML.crust.operation.baseoperation import BaseOperation, SplitOperation

class TestBaseOperation(TestBase):

    def test_BaseOperation(self):
        pass

    def test_SplitOperation(self):
        wave_rng = [2, 8]
        OP = SplitOperation(wave_rng)
        wave_new = OP.perform(np.arange(10))
        self.assertTrue(wave_new.min() >= wave_rng[0])
        self.assertTrue(wave_new.max() <= wave_rng[1])
        self.assertEqual(OP.split_idxs.shape, (2,))




