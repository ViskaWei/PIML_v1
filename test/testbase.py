import os
import numpy as np
import unittest
from PIML.crust.data.constants import Constants
from PIML.crust.data.spec.basespec import StellarSpec

from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.gateway.loaderIF.baseloaderIF import ObjectLoaderIF, SkyLoaderIF, SpecGridLoaderIF, SpecLoaderIF 

GRID_PATH="/datascope/subaru/user/swei20/data/pfsspec/import/stellar/grid"
DATA_DIR ="/home/swei20/PIML_v1/"

class TestBase(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
    



        self.test_DATA_PATH = DATA_DIR + "test/testdata/bosz_5000_test.h5"

        self.DATA_PATH=os.path.join(GRID_PATH, "bosz_5000_RHB.h5")

        SGL = SpecGridLoaderIF()
        SGL.set_data_path(self.DATA_PATH)
        self.specGrid = SGL.load()

        self.wave =  self.specGrid.wave      #3000-14000, 15404
        self.flux =  self.specGrid.flux      #(2880, 15404)
        self.para = self.specGrid.coord      #(2880, 5)
        self.pdx  = self.specGrid.coord_idx  #(2880, 5)
        self.pdx0 = self.pdx - self.pdx[0]   #(2880, 5)

        SL = SpecLoaderIF()
        SL.set_data_path(self.DATA_PATH)
        self.spec = SL.load()

        self.SKY_PATH = DATA_DIR +"test/testdata/wavesky.npy"
        self.Sky  = SkyLoaderIF().load(self.SKY_PATH)
        self.skyH = np.load(DATA_DIR + "test/testdata/skyH.npy")  #2204
        self.sky = np.load(DATA_DIR +"test/testdata/sky.npy")    #220

        self.waveH_RedM = np.load(DATA_DIR +"/test/testdata/waveH_RedM.npy")
        self.wave_RedM  = np.load(DATA_DIR +"/test/testdata/wave_RedM.npy")
        self.fluxH_mid  = np.load(DATA_DIR +"/test/testdata/fluxH_mid.npy") 


        self.OBJECT = {"DATA_PATH": self.DATA_PATH}

        self.OP_DATA = {"SKY_PATH": self.SKY_PATH}

        self.arm = "RedM"
        self.OP_PARAMS = {
            "box_name": "R",
            "arm": self.arm,
            "step": 10,
            "wave_rng": Constants.ARM_RNGS[self.arm]
        }

        self.OP_MODELS = {
            "Resolution": "Alex",
            "Interp": "RBF",
        }

        self.PARAMS = {
            "object": self.OBJECT,
            "data":   self.OP_DATA,
            "op"  :   self.OP_PARAMS,
            "model":  self.OP_MODELS,
        }

        

        
    # def get_TestSpecGrid():
    #     xgrid = np.mgrid[0:5,0:5]
    #     coordx = xgrid.reshape(2, -1).T
    #     value_1D  = xgrid[0].flatten()
    #     value_2D  = np.tile(value_1D, (2,1)).T
    #     coordx_to_interp = np.array([[1.5,1.5], [2,2]])
    #     specGrid = StellarSpecGrid(value_1D, value_2D, coordx, None)


    def get_SpecGrid(self):
        specGrid = StellarSpecGrid(self.wave, self.flux, self.para, self.pdx)   
        return specGrid

    def get_Spec(self):
        spec     = StellarSpec(self.wave, self.flux)
        return spec

    def same_array(self, a, b):
        return self.assertIsNone(np.testing.assert_array_equal(a, b))
        