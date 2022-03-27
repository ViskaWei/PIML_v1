import os

import numpy as np

from unittest import TestCase
from PIML.crust.model.interpmodel import RBFInterpBuilderModel

DATA_DIR= "/home/swei20/PIML_v1/test/testdata/testmethoddata/"

class TestInterpModel(TestCase):

    def test_RBFInterpBuilderModel(self):
        coordx, value_2D = self.get_test_data()
        model = RBFInterpBuilderModel()
        interpolator = model.apply(coordx, value_2D)

        self.check_build(interpolator)
        self.check_store(model, name="test_RBFInterpBuilder")
        

    def get_test_data(self):
        xgrid = np.mgrid[0:5,0:5]
        coordx = xgrid.reshape(2, -1).T
        value_1D  = xgrid[0].flatten()
        return coordx, value_1D
        
    def check_store(self, model, name = "interp"):
        PATH = os.path.join(DATA_DIR, name + ".pickle")
        if os.path.exists(PATH): os.remove(PATH)

        model.store(DATA_DIR, name)
        self.assertTrue(os.path.exists(PATH))

    def check_build(self, interpolator):
        coordx_to_interp = np.array([1.5, 1.5])
        value_to_check   = np.array([1.5, 1.5])
        value_interped   = interpolator(coordx_to_interp).round(1)
        self.assertIsNone(np.testing.assert_array_almost_equal(value_interped, value_to_check))


    

