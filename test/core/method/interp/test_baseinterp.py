import numpy as np
from unittest import TestCase
from PIML.core.method.interp.baseinterp import BaseInterp, RBFInterp
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF

class TestBaseInterp(TestCase):
    """
    Test the BaseInterpModel class.
    """
    def test_RBFInterp(self):
        PATH = "test/testdata/testmethoddata/interp.pickel"
        pass

