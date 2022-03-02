import numpy as np
from PIML.surface.database.baseloader import H5pyLoader, ZarrLoader
from test.testbase import TestBase



class TestBaseLoader(TestBase):


    def test_H5pyLoader(self):
        
        loader = H5pyLoader()
        DArgs = loader.load_DArgs(self.DATA_PATH)
        self.assertIsNotNone(DArgs)
        self.assertEqual(DArgs["flux"].shape, (120,1178))
        self.assertEqual(DArgs["wave"].shape, (1178,))
        self.assertIsNone(np.testing.assert_array_equal(DArgs["pdx"][0], [6, 8, 4, 3, 1]))

        
        