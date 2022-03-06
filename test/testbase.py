import numpy as np
from unittest import TestCase
from PIML.gateway.loaderIF.baseloaderIF import ObjectLoaderIF, SpecGridLoaderIF, SpecLoaderIF 

class TestBase(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

        self.DATA_PATH = "test/testdata/bosz_5000_test.h5"
        
        self.loaderIF = ObjectLoaderIF()
        self.loaderIF.set_data_path(self.DATA_PATH)

        self.wave = self.loaderIF.load_arg("wave")
        self.flux = self.loaderIF.load_arg("flux")
        self.para = self.loaderIF.load_arg("para")
        self.pdx  = self.loaderIF.load_arg("pdx")

        SGL = SpecGridLoaderIF()
        SGL.set_data_path(self.DATA_PATH)
        self.specGrid = SGL.load()

        SL = SpecLoaderIF()
        SL.set_data_path(self.DATA_PATH)
        self.spec = SL.load()

    

        self.params = {
            "DATA_PATH": self.DATA_PATH,
            "wRng": [8200.0, 8400.0], #len(wave) = 241 out of 1178
        }

        self.MODEL_TYPES = {
            "ResTunableProcess": "Alex"
        }


    