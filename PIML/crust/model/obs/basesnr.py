import numpy as np
import scipy as sp
import logging
from abc import ABC, abstractmethod
from PIML.crust.data.spec.obs.baseobs import Obs

class BaseSNR(ABC):
    @abstractmethod
    def convert():
        pass
    @abstractmethod
    def get_snr():
        pass

class BaseSNRNL(BaseSNR):
    '''
    snr to noise level converter
    '''
    def __init__(self, Obs:Obs) -> None:
        self.Obs = Obs
        self.noise_level_grid = [10, 30, 40, 50, 100, 200, 300, 400, 500]

    def create_snr_nl_converter(self, flux, sky, avg=10):
        assert flux.shape[-1] == sky.shape[0]
        sigma = self.Obs.simulate_sigma(flux, sky)
        SN = self.average_converter(avg, flux, sigma)
        logging.info(f"snr2nl-SN: {SN}")

        f = sp.interpolate.interp1d(SN, self.noise_level_grid, fill_value=0)
        f_inv = sp.interpolate.interp1d(self.noise_level_grid, SN, fill_value=0)
        return f, f_inv

    def convert_nl2snr(self, flux, sigma):
        noise = self.Obs.get_noise(sigma)
        SN = []
        for noise_level in self.noise_level_grid:
            obsfluxs = flux + noise_level * noise
            sn = self.Obs.get_snr(obsfluxs, sigma, noise_level)
            SN.append(sn)
        return SN

    def average_converter(self, avg, *args):
        SNS=[]
        for i in range(avg):
            SNS.append(self.convert_nl_to_snr(*args))
        SN = np.mean(SNS, axis=0)
        return SN
