import os
import numpy as np
from unittest import TestCase
from PIML.crust.data.constants import Constants
from PIML.crust.data.specdata.basespec import StellarSpec
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.gateway.loaderIF.baseloaderIF import InterpLoaderIF, ObjectLoaderIF, SkyLoaderIF, SpecGridLoaderIF, SpecLoaderIF 

GRID_PATH="/datascope/subaru/user/swei20/data/pfsspec/import/stellar/grid"
ROOT = "/home/swei20/PIML_v1/"

SPEC_GRID_DATA_DIR = os.path.join(ROOT, "test/testdata/testspecgriddata/")
NN_PREP_DATA_DIR = os.path.join(ROOT, "test/testdata/testnnprepdata/")

class DataInitializer():
    def __init__(self):
        self.DATA_PATH=os.path.join(GRID_PATH, "bosz_5000_RHB.h5")
        self.set_SpecGrid_data()
        self.set_NNPrep_data()
        self.PARAMS = {
            "SpecGrid": self.SPEC_GRID_PARAMS,
            "NNPrep": self.NN_PREP_PARAMS,
        }

    def set_SpecGrid_data(self, DATA_DIR=SPEC_GRID_DATA_DIR):
        self.SpecGrid_TEST_PATH = DATA_DIR + "bosz_5000_test.h5"

        SGL = SpecGridLoaderIF()
        SGL.set_path(self.DATA_PATH)
        self.specGrid = SGL.load()

        self.wave =  self.specGrid.wave      #3000-14000, 15404
        self.flux =  self.specGrid.flux      #(2880, 15404)
        self.para = self.specGrid.coord      #(2880, 5)
        self.pdx  = self.specGrid.coord_idx  #(2880, 5)
        self.pdx0 = self.pdx - self.pdx[0]   #(2880, 5)
        self.midx = 1377

        SL = SpecLoaderIF()
        SL.set_path(self.DATA_PATH)
        self.spec = SL.load()

        self.SKY_PATH = DATA_DIR +"wavesky.npy"
        self.Sky  = SkyLoaderIF().load(self.SKY_PATH)
        self.skyH = np.load(DATA_DIR + "skyH.npy")  #2204
        self.sky = np.load(DATA_DIR +"sky.npy")    #220

        self.waveH_RedM = np.load(DATA_DIR +"waveH_RedM.npy")
        self.wave_RedM  = np.load(DATA_DIR +"wave_RedM.npy")
        self.fluxH_mid  = np.load(DATA_DIR +"fluxH_mid.npy") 
        self.flux_mid   = np.load(DATA_DIR + "flux_mid.npy")
        self.sigma_mid  = np.load(DATA_DIR + "sigma_mid.npy")
        self.coord_interp = np.array([-0.5,  6125,  2.5, -0.25,  0.0])
        self.coordx_interp = np.array([2., 2.5, 1., 2., 1.])

        self.logflux_interp = np.load(DATA_DIR + "logflux_interp.npy")

        self.OBJECT = {"DATA_PATH": self.DATA_PATH}

        self.OP_DATA = {"SKY_PATH": self.SKY_PATH, "Sky": self.Sky}

        self.arm = "RedM"
        self.step = 10
        self.box_name = "R"

        self.OP_PARAMS = {
            "box_name": self.box_name,
            "arm": self.arm,
            "step": self.step,
            "wave_rng": Constants.ARM_RNGS[self.arm]
        }

        self.OP_MODELS = {
            "Resolution": "Alex",
            "Interp": "RBF",
        }

        self.SPEC_GRID_PARAMS = {
            "object": self.OBJECT,
            "data":   self.OP_DATA,
            "op"  :   self.OP_PARAMS,
            "model":  self.OP_MODELS,
        }





    def set_NNPrep_data(self, DATA_DIR=NN_PREP_DATA_DIR):
        loader = InterpLoaderIF()
        self.RBFinterp = loader.load(DATA_DIR + "interp.pickle")

        self.NN_PREP_PARAMS = {
            "rng": np.array([4., 5., 3., 5., 3.]),
        }




class TestBase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.D = DataInitializer()
        

    # def get_TestSpecGrid():
    #     xgrid = np.mgrid[0:5,0:5]
    #     coordx = xgrid.reshape(2, -1).T
    #     value_1D  = xgrid[0].flatten()
    #     value_2D  = np.tile(value_1D, (2,1)).T
    #     coordx_to_interp = np.array([[1.5,1.5], [2,2]])
    #     specGrid = StellarSpecGrid(value_1D, value_2D, coordx, None)


    def get_SpecGrid(self):
        specGrid = StellarSpecGrid(self.D.wave, self.D.flux, self.D.para, self.D.pdx)   
        return specGrid

    def get_Spec(self):
        Spec     = StellarSpec(self.D.wave, self.D.flux)
        return Spec

    def get_Grid(self):
        Grid = StellarGrid(self.D.para, self.D.pdx)
        return Grid


    def same_array(self, a, b):
        return self.assertIsNone(np.testing.assert_array_equal(a, b))
        
    def close_array(self, a, b, tol=1e-3):
        return self.assertIsNone(np.testing.assert_allclose(a, b, atol=tol))

    def check_StellarSpec(self, Spec: StellarSpec):
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
        # if SpecOnly:
        #     nl_to_check  = np.array([152., 76., 46.])
        #     snr_to_check = np.array([135., 90., 45.])
        # else:
        nl_to_check  = np.array([157., 78., 47.])
        snr_to_check = np.array([140., 93., 47.])

        self.assertIsNone(np.testing.assert_allclose(
            Spec.map_snr    ([10, 20, 30]), nl_to_check,
            atol=1))
        self.assertIsNone(np.testing.assert_allclose(
            Spec.map_snr_inv([10, 20, 30]), snr_to_check,
            atol=1))
        
        # AddPfsObsSpecOperation
        self.assertEqual(Spec.Obs.step, self.D.step)
        self.same_array(Spec.Obs.sky, self.D.sky)
        self.same_array(Spec.Obs.cal_sigma(flux_mid), self.D.sigma_mid)

        # LogSpecOperation
        self.same_array(Spec.logflux[self.D.midx], np.log(self.D.flux_mid))

    def check_StellarSpecGrid(self, SpecGrid: StellarSpecGrid):
        self.check_StellarSpec(SpecGrid)
        # InterpSpecGridOperation
        logflux_interp = SpecGrid.interpolator(self.D.coord_interp,  scale=True)
        self.close_array(logflux_interp, self.D.logflux_interp, tol=1e-4)
        logflux_interp2 = SpecGrid.interpolator(self.D.coordx_interp, scale=False)
        self.same_array(logflux_interp, logflux_interp2)
        self.close_array(logflux_interp, self.D.logflux_interp, tol=1e-4)
        





if __name__ == "__main__":
    TestCase.main()