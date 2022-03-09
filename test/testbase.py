from ast import Constant
import os
import numpy as np
from unittest import TestCase
from PIML.crust.data.constants import Constants
from PIML.crust.data.spec.basespec import StellarSpec

from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.gateway.loaderIF.baseloaderIF import ObjectLoaderIF, SpecGridLoaderIF, SpecLoaderIF 

GRID_PATH="/datascope/subaru/user/swei20/data/pfsspec/import/stellar/grid"

class TestBase(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.test_DATA_PATH = "test/testdata/bosz_5000_test.h5"

        self.DATA_PATH=os.path.join(GRID_PATH, "bosz_5000_RHB.h5")

        SGL = SpecGridLoaderIF()
        SGL.set_data_path(self.DATA_PATH)
        self.specGrid = SGL.load()

        self.wave =  self.specGrid.wave
        self.flux =  self.specGrid.flux
        self.para = self.specGrid.coord
        self.pdx  = self.specGrid.coord_idx
        self.pdx0 = self.pdx - self.pdx[0]

        SL = SpecLoaderIF()
        SL.set_data_path(self.DATA_PATH)
        self.spec = SL.load()

        self.arm = "RedM"
        self.DATA_PARAMS = {"DATA_PATH": self.DATA_PATH},
        self.OP_PARAMS = {
            "box_name": "R",
            "arm": self.arm,
            "step": 10,
            "wave_rng": Constants.ARM_RNGS[self.arm]
        }

        self.PARAMS = {
            "data": self.DATA_PARAMS,
            "op"  : self.OP_PARAMS,
        }

        

        self.MODEL_TYPES = {
            "Resolution": "Alex"
        }


    def get_SpecGrid(self):
        specGrid = StellarSpecGrid(self.wave, self.flux, self.para, self.pdx)   
        return specGrid

    def get_Spec(self):
        spec     = StellarSpec(self.wave, self.flux)
        return spec