import numpy as np
from test.testbase import TestBase
from PIML.crust.process.baseprocess import StellarSpecProcess

class TestBaseProcess(TestBase):
    def test_BaseProcess(self):
        pass

    def test_StellarSpecProcess(self):
        Spec = self.get_Spec()
        Process = StellarSpecProcess()
        Process.set_process(self.D.OP_PARAMS, self.D.OP_MODELS, self.D.OP_DATA)
        Process.start(Spec)
        
        flux_mid = Spec.flux[self.D.midx]
        
        # SplitSpecOperation, TuneSpecOperation
        #TODO fix this wave
        self.assertIsNone(np.testing.assert_allclose(
            Spec.wave, self.D.wave_RedM, 
            atol=1e-3))
        self.same_array(flux_mid, self.D.flux_mid)
        # SimulateSkySpecOperation
        self.same_array(Spec.sky, self.D.sky)
        self.same_array(Spec.skyH, self.D.skyH)
        # MapSNRSpecOperation
        self.assertIsNone(np.testing.assert_allclose(
            Spec.map_snr    ([10, 20, 30]), np.array([152, 76, 46]),
            atol=1))
        self.assertIsNone(np.testing.assert_allclose(
            Spec.map_snr_inv([10, 20, 30]), np.array([135, 90, 45]),
            atol=1))
        
        # AddPfsObsSpecOperation
        self.assertEqual(Spec.Obs.step, self.D.step)
        self.same_array(Spec.Obs.sky, self.D.sky)
        self.same_array(Spec.Obs.cal_sigma(flux_mid), self.D.sigma_mid)

        # LogSpecOperation
        self.same_array(Spec.logflux[self.D.midx], np.log(self.D.flux_mid))

def main():de

        



