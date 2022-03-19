import numpy as np
from unittest import TestCase

from PIML.crust.model.specgrid.interpspecgridmodel import RBFInterpBuilder, RBFInterpSpecGridModel


class TestInterpSpecGridModel(TestCase):
    """
    Test the BaseInterpModel class.
    """
    def test_RBFInterpBuilder(self):
        coordx, value_2D = self.get_test_data()
        builder = RBFInterpBuilder()        
        interpolator = builder.build(coordx, value_2D)
        self.check_base_interpolator(interpolator)

    def test_InterpSpecGridModel(self):
        coordx, value_2D = self.get_test_data()
        model = RBFInterpSpecGridModel()
        model.set_model_param(kernel="gaussian", epsilon=0.5)
        model.set_model_data(coordx, value_2D)
        
        self.check_base_interpolator(model.base_interpolator)

    def test_InterpSpecGridModel_on_SpecGrid(self):
        pass


    def check_base_interpolator(self, interpolator):
        coordx_to_interp = np.array([[1.5, 1.5], [2, 2]])
        value_to_check   = np.array([[1.5, 1.5], [2, 2]])
        value_interped_1d   = interpolator(coordx_to_interp[0]).round(1)
        self.assertIsNone(np.testing.assert_array_almost_equal(value_interped_1d, value_to_check[0]))

        value_interped   = interpolator(coordx_to_interp).round(1)
        self.assertIsNone(np.testing.assert_array_almost_equal(value_interped, value_to_check))

    def get_test_data(self):
        xgrid = np.mgrid[0:5,0:5]
        coordx = xgrid.reshape(2, -1).T
        value_1D  = xgrid[0].flatten()
        value_2D  = np.tile(value_1D, (2,1)).T
        return coordx, value_2D

