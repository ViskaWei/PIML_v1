import numpy as np
from unittest import TestCase
from PIML.core.method.interp.baseinterp import BaseInterp, RBFInterp
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF, FileLoaderIF

class TestBaseInterp(TestCase):
    """
    Test the BaseInterpModel class.
    """
    def test_RBFInterp(self):
        PATH = "test/testdata/testmethoddata/interp.pickel"
        pass
        # loader = FileLoaderIF()
        # self.RBFinterp = loader.load(DATA_DIR + "interp.pickle")


        # interpolator = builder.build(coordx, value_2D)
        # self.check_base_interpolator(interpolator)

    # def test_InterpSpecGridModel(self):
    #     coordx, value_2D = self.get_test_data()
    #     model = RBFInterpSpecGridModel()
    #     model.set_model_param(kernel="gaussian", epsilon=0.5)
    #     model.set_model_data(coordx, value_2D)
        
    #     self.check_base_interpolator(model.base_interpolator)

    # def test_InterpSpecGridModel_on_SpecGrid(self):
    #     pass


