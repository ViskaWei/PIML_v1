import numpy as np
from unittest import TestCase

from PIML.crust.model.interp.baseinterpmodel import RBFInterpModel


class TestBaseInterpModel(TestCase):
    """
    Test the BaseInterpModel class.
    """
    
    def test_RBFInterpModel(self):
        xgrid = np.mgrid[0:5,0:5]
        coordx = xgrid.reshape(2, -1).T
        value_1D  = xgrid[0].flatten()
        value_2D  = np.tile(value_1D, (2,1)).T
        coordx_to_interp = np.array([[1.5,1.5], [2,2]])

        model = RBFInterpModel()        
        self.assertEqual(model.name, "RBF")
        model.set_model_param()

        interpolator = model.apply(coordx, value_1D)
        value_interped   = interpolator(coordx_to_interp).round(1)
        self.assertIsNone(np.testing.assert_array_almost_equal(value_interped, np.array([1.5,2])))

        interpolator = model.apply(coordx, value_2D)
        value_interped   = interpolator(coordx_to_interp).round(1)
        self.assertIsNone(np.testing.assert_array_almost_equal(value_interped, np.array([[1.5, 1.5], [2,2]])))

