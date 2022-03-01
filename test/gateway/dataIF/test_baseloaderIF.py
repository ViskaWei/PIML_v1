import numpy as np
from PIML.gateway.dataIF.baseloaderIF import BaseLoaderIF
from PIML.surface.database.baseloader import H5pyLoader
from test.testbase import TestBase


DATA_PATH = "test/testdata/bosz_5000_test.h5"

class TestBaseLoader(TestBase):



    def test_BaseLoaderIF(self):
        loader = H5pyLoader()

        loaderIF = BaseLoaderIF()
        DArgs = loaderIF.set
        self.assertIsNotNone(DArgs)
        self.assertEqual(DArgs["flux"].shape, (120,1178))
        self.assertEqual(DArgs["wave"].shape, (1178,))
        self.assertIsNone(np.testing.assert_array_equal(DArgs["pdx"][0], [6, 8, 4, 3, 1]))

        
        