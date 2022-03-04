import numpy as np

from .baseprocessIF import BasePrepareIF
from ..loaderIF.baseloaderIF import BaseLoaderIF, WaveLoaderIF







class WavePrepareIF(BasePrepareIF):
    """ class for separate data into detector arms. """
    def __init__(self) -> None:
        super().__init__()
        self.startIdx = None
        self.endIdx = None
        

    def set_param(self, param):
        super().set_param(param)
        self.wRng = param["wRng"]

    def set_data(self, data):
        super().set_data(data)

    def set_data_from_(self, loaderIF: BaseLoaderIF):
        pass

    def process_data(self):
        self.get_wave_in_wRng(self.wave, self.wRng)

    def get_wave_in_wRng(self, wave, wRng):
        # get wave in waveRng
        startIdx, endIdx = np.digitize(wRng, wave)
        wave_processed = wave[startIdx:endIdx]
        return wave_processed, (startIdx, endIdx)

    def prepare(self, param, data):
        super().prepare(param, data)