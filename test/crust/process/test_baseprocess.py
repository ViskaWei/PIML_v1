import numpy as np
from test.testbase import TestBase
from PIML.crust.process.baseprocess import StellarSpecProcess

class TestBaseProcess(TestBase):
    def test_BaseProcess(self):
        pass

    def test_StellarProcess(self):
        self.check_StellarSpecProcess()

    def check_StellarSpecProcess(self):
        Spec = self.get_Spec()
        Process = StellarSpecProcess()
        Process.set_process(self.D.OP_PARAMS, self.D.OP_MODELS, self.D.OP_DATA)
        Process.start(Spec)
        # SplitSpecOperation, TuneSpecOperation
        self.same_array(Spec.wave, self.D.wave_RedM)
        self.same_array(Spec.flux[self.D.midx], self.D.flux_mid)
        # SimulateSkySpecOperation
        self.same_array(Spec.sky, self.D.sky)
        self.same_array(Spec.skyH, self.D.skyH)
        # MapSNRSpecOperation
        self.assertIsNone(np.testing.assert_allclose(Spec.map_snr    ([10, 20, 30]).round(),  np.array([152, 76, 46])))
        self.assertIsNone(np.testing.assert_allclose(Spec.map_snr_inv([10, 20, 30]).round(),  np.array([135, 90, 45])))
        
        # AddPfsObsSpecOperation
        self.Spec.Obs

        # LogSpecOperation
        self.same_array(Spec.logflux[self.D.midx], np.log(self.D.flux_mid))

        

        



