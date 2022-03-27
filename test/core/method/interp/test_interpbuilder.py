import numpy as np
from unittest import TestCase
from PIML.core.method.interp.interpbuilder import RBFInterpBuilder

class TestInterpBuilder(TestCase):

    def get_test_data(self):
        xgrid = np.mgrid[0:5,0:5]
        coordx = xgrid.reshape(2, -1).T
        value_1D  = xgrid[0].flatten()
        value_2D  = np.tile(value_1D, (2,1)).T
        return coordx, value_2D

    def test_RBFInterpBuilder(self):
        coordx, value_2D = self.get_test_data()
        builder = RBFInterpBuilder("gaussian", 0.5)
        
        builder.build(coordx, value_2D)
        self.check_build(builder.interpolator)


    def check_build(self, interpolator):
        coordx_to_interp = np.array([[1.5, 1.5], [2, 2]])
        value_to_check   = np.array([[1.5, 1.5], [2, 2]])

        value_interped   = interpolator(coordx_to_interp).round(1)
        self.assertIsNone(np.testing.assert_array_almost_equal(value_interped, value_to_check))


    

