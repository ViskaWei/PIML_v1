
import numpy as np
from test.testbase import TestBase
from PIML.crust.operation.specoperation import BaseSpecOperation, \
    LogSpecOperation, SplitSpecOperation, TuneSpecOperation,\
    SimulateSkySpecOperation, MapSNRSpecOperation
from PIML.crust.model.spec.resolutionmodel import ResolutionModel


class TestSpecOperation(TestBase):

    def test_LogSpecOperation(self):
        flux = np.arange(1,10)
        logflux_to_check = np.log(flux)
        OP = LogSpecOperation()
        logflux = OP.perform(flux)
        self.assertIsNone(np.testing.assert_array_equal(logflux, logflux_to_check))

    def test_SimulateSkySpecOperation(self):

        OP = SimulateSkySpecOperation(self.Sky)
        sky = OP.perform(self.waveH_RedM)
        sky_to_check = self.skyH
        self.assertIsNone(np.testing.assert_array_equal(sky, sky_to_check))

    def test_MapSNRSpecOperation(self):
        OP = MapSNRSpecOperation()
        map_snr, map_snr_inv = OP.perform(self.fluxH_mid, self.skyH)
        self.assertIsNone(np.testing.assert_allclose(map_snr([10,20,30]).round(),  np.array([152,  76,  46])))


    def test_SplitSpecOperation(self):
        
        OP = SplitSpecOperation(self.arm)
        
        wave_new = OP.perform(self.wave)
        self.assertIsNotNone(wave_new)

        Spec = self.check_perform_on_spec(OP) 
        self.assertIsNone(np.testing.assert_array_less(OP.rng[0], Spec.wave))
        self.assertIsNone(np.testing.assert_array_less(Spec.wave, OP.rng[1]))
        if self.arm =="RedM":
            self.same_array(wave_new, self.waveH_RedM)
        self.assertEqual(OP.split_idxs.shape, (2,))
        
        # sky test cannot be done on full spec
        OP = SimulateSkySpecOperation(self.Sky)
        OP.perform_on_Spec(Spec)
        self.assertIsNotNone(Spec.sky)




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


