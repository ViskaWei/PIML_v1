from PIML.gateway.dataloader.baseloader import h5pyLoader, zarrLoader
from test.testbase import TestBase


class TestBaseLoader(TestBase):

    def test_h5pyloader(self):
        
    
        loader = h5pyLoader()
        self.DATA_PATH = "test/testdata/bosz_5000_test.h5"

        DArgs = loader.load_DArgs(self.DATA_PATH)
        self.assertIsNotNone(DArgs)
        # self.assertEqual(DArgs["flux"].shape, (120,1178))
        # self.assertEqual(DArgs["wave"].shape, (1178,))
        # self.assertEqual(DArgs["pdx"][0], [6, 8, 4, 3, 1])

        
        