import numpy as np
from unittest import TestCase

class TestBase(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.DATA_PATH = "test/testdata/bosz_5000_test.h5"
        self.wave = np.load("test/testdata/wave.npy")
        self.flux = np.load("test/testdata/flux.npy")
        self.para = np.load("test/testdata/para.npy")
        self.pdx  = np.load("test/testdata/pdx.npy")

        self.params = {
            "DATA_PATH": self.DATA_PATH,
            "wRng": [8200.0, 8400.0], #len(wave) = 241 out of 1178
        }



    