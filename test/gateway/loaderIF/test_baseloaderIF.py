import pandas as pd
from PIML.gateway.loaderIF.baseloaderIF import ObjectLoaderIF, SpecGridLoaderIF, SpecLoaderIF, BoxParamLoaderIF
from test.testbase import TestBase

class TestBaseLoader(TestBase):

    def test_SpecGridLoaderIF(self):        
        loaderIF = SpecGridLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        specGrid = loaderIF.load()

        self.assertIsNotNone(specGrid.wave)
        self.assertEqual(specGrid.wave.shape, (1178,))

        self.assertIsNotNone(specGrid.flux)
        self.assertEqual(specGrid.flux.shape, (120, 1178))

        self.assertIsNotNone(specGrid.para)
        self.assertEqual(specGrid.para.shape, (120,5))

        self.assertIsNotNone(specGrid.pdx)
        self.assertEqual(specGrid.pdx.shape, (120,5))
        
    def test_SpecLoaderIF(self):
        loaderIF = SpecLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        spec = loaderIF.load()

        self.assertIsNotNone(spec.wave)
        self.assertEqual(spec.wave.shape, (1178,))

        self.assertIsNotNone(spec.flux)
        self.assertEqual(spec.flux.shape, (120, 1178))

        
    def test_BoxParamLoaderIF(self):
        loaderIF = BoxParamLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        boxParam = loaderIF.load()

        self.assertIsNotNone(boxParam.para)
        self.assertIsNotNone(boxParam.pdx)
        self.assertIsNotNone(boxParam.PhyShort)

        boxParam.set_dfpara()
        self.assertIsNotNone(boxParam.dfpara)
        self.assertIsInstance(boxParam.dfpara, pd.DataFrame)

        self.assertEqual(boxParam.para.shape, (120,5))


    def test_BaseLoaderIF(self):
        for LoaderIF in ObjectLoaderIF.__subclasses__():
            loaderIF = LoaderIF()
            loaderIF.set_data_path(self.DATA_PATH)
            data = loaderIF.load()
            self.assertIsNotNone(data)