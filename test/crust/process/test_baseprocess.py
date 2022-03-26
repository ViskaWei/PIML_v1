import numpy as np
from test.testbase import TestBase
from PIML.crust.process.baseprocess import StellarSpecProcess

class TestBaseProcess(TestBase):
    def test_BaseProcess(self):
        pass

    def test_StellarSpecProcess(self):
        Spec = self.get_Spec()
        np.random.seed(922)

        Process = StellarSpecProcess()
        Process.set_process(self.D.OP_PARAM, self.D.OP_MODEL, self.D.OP_DATA)
        Process.start(Spec)

        self.check_StellarSpec(Spec)
        
        







        



