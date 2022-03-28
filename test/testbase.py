import os
import numpy as np
from unittest import TestCase
from PIML.crust.data.nndata.baseprepnn import PrepNN
from PIML.crust.data.specdata.basespec import StellarSpec
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from test.testdata.datainitializer import DataInitializer

# GRID ="/datascope/subaru/user/swei20/data/pfsspec/import/stellar/grid"
# ROOT = "/home/swei20/PIML_v1/"



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

    def get_PrepNN(self):
        PrepNN(self.D.RBFinterp, self.D.sky, self.D.arm, self.D.res)
        return PrepNN

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
        self.same_array(Spec.Obs.get_sigma(flux_mid), self.D.sigma_mid)

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
        
    def check_PrepNN(self, PrepNN: PrepNN):
        self.assertTrue(PrepNN.name == self.D.name)
        
        
        #finishPredNNOperation
        self.assertTrue(PrepNN.train_name == (self.D.name + f"_N{self.D.ntrain}_train"))
        self.assertTrue(PrepNN.test_name  == (self.D.name + f"_N{self.D.ntest}_test"))






if __name__ == "__main__":
    TestCase.main()