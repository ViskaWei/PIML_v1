from abc import abstractmethod
import numpy as np
from PIML.crust.model.spec.basespecmodel import BaseSpecModel
from PIML.crust.data.spec.basespec import StellarSpec

class ResolutionModel(BaseSpecModel):
    @abstractmethod
    def tune_resolution(self, ):
        pass

class AlexResolutionModel(ResolutionModel):
    """ class for tuning data with Alex method """

    @property
    def name(self):
        return "Alex"
    
    def apply(self, data):
        return self.tune_resolution(data, self.step)

    def apply_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        Spec.wave = self.apply(Spec.wave)
        Spec.flux = self.apply(Spec.flux)
        if hasattr(Spec, "sky"):
            Spec.skyH = Spec.sky
            Spec.sky = self.apply(Spec.sky)


    def set_model_param(self, step):
        self.step = step

    def tune_resolution(self, data, step):
        data_cumsumed = np.cumsum(data, axis=-1)
        data_cumsumed_sampled_every_step = data_cumsumed[..., ::step]
        data_sampled_every_step = np.diff(data_cumsumed_sampled_every_step, axis=-1)
        data_averaged_every_step = np.divide(data_sampled_every_step, step)
        return data_averaged_every_step


class NpResolutionModel(ResolutionModel):
    #TODO: implement this
    """ class for tuning data with Numpy method """

    @property
    def name(self):
        return "Np"
    
    def apply(self):
        pass

    def apply_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        pass

    def set_model_param(self, param):
        self.wave = param["wave"]
        self.wref = param["wref"]
        self.res_in = param["res_in"]
        self.res_out = param["res_out"]

    def get_sigmas(self):
        sigma_input = self.wref / self.res_in
        sigma_output = self.wref / self.res_out
        self.sigma_kernel = np.sqrt(sigma_output**2 - sigma_input**2)

    def get_kernel(self):
        kernel_mask = (self.wref - 5 < self.wave) & (self.wave < self.wref + 5)
        kernel_wave = self.wave[kernel_mask][:-1]
        kernel_wave -= kernel_wave[kernel_wave.size // 2]
        # print(kernel_wave.shape, kernel_wave)
        kernel = self.gauss_kernel(kernel_wave, self.sigma_kernel)
        self.kernel = kernel / kernel.sum()

    def gauss_kernel(self, dwave, sigma):
        return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-dwave**2 / (2 * sigma**2))

    def tune_resolution(self, data, step):
        self.get_sigmas()
        self.get_kernel()

        data_tuned = np.convolve(self.kernel, data, mode='same')
        return data_tuned


    @staticmethod
    def correct_wave_grid(wlim, resolution):
    # BOSZ spectra are written to the disk with 3 decimals which aren't
    # enough to represent wavelength at high resolutions. This code is
    # from the original Kurucz SYNTHE to recalculate the wavelength grid.

        RESOLU = resolution
        WLBEG = wlim[0]  # nm
        WLEND = wlim[1]  # nm
        RATIO = 1. + 1. / RESOLU
        RATIOLG = np.log10(RATIO)
        
        IXWLBEG = int(np.round(np.log10(WLBEG) / RATIOLG))
        WBEGIN = 10 ** (IXWLBEG * RATIOLG)
        if WBEGIN < WLBEG:
            IXWLBEG = IXWLBEG + 1
            WBEGIN = 10 ** (IXWLBEG * RATIOLG)
            
        IXWLEND = int(np.round(np.log10(WLEND) / RATIOLG))
        WLLAST = 10 ** (IXWLEND * RATIOLG)
        if WLLAST >= WLEND:
            IXWLEND = IXWLEND - 1
            WLLAST = 10 ** (IXWLEND * RATIOLG)
        LENGTH = IXWLEND - IXWLBEG + 1
        DWLBEG = WBEGIN * RATIO - WBEGIN
        DWLLAST = WLLAST - WLLAST / RATIO
        
        a = np.linspace(np.log10(WBEGIN), np.log10(WLLAST), LENGTH)
        cwave = 10 ** a
        
        return cwave

