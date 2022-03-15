
import numpy as np
from test.testbase import TestBase
from PIML.crust.model.obs.baseobs import Obs

class TestBaseObs(TestBase):
    def test_Obs(self):
        flux = np.arange(10) + 10
        sky = np.arange(10) +1
        var_to_check = np.array([16020., 16031., 16042., 16053., 16064.,\
                                16075., 16086., 16097., 16108., 16119.])
        obs = Obs()
        var = obs.get_var(flux, sky)
        self.assertIsNone(np.testing.assert_array_equal(var, var_to_check))
        
        sigma = obs.simulate_sigma(flux, sky)
        sigma_to_check =  np.ones(10) * 127.
        self.assertIsNone(np.testing.assert_array_equal(sigma.round(), sigma_to_check))

        fluxs = np.tile(flux, (2,1))
        vars = obs.get_var(fluxs, sky)
        vars_to_check = np.tile(var_to_check, (2,1))
        self.assertIsNone(np.testing.assert_array_equal(vars, vars_to_check))

        self.check_get_snr(obs)
        

    def check_get_snr(self, obs):
        sigma = np.arange(1, 10)
        noise_to_check = np.array([ -1.38,  -1.89,   0.15,   3.78,  -2.98,\
                                        -10.87,  -4.89,  -4.96, -1.74])
        np.random.seed(922)
        noise = obs.get_noise(sigma).round(2)
        self.assertIsNone(np.testing.assert_array_equal(noise, noise_to_check))

        snr = obs.get_snr(sigma+noise, sigma).round(2)
        snr_to_check = 0.42
        self.assertEqual(snr, snr_to_check)

        snr = obs.get_snr(sigma+noise, sigma, noise_level=10).round(2)
        snr_to_check = 0.04
        self.assertEqual(snr, snr_to_check)
        



