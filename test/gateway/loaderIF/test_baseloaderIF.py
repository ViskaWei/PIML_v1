import pandas as pd
from PIML.gateway.loaderIF.baseloaderIF import ObjectLoaderIF, WaveLoaderIF, FluxLoaderIF, SpecLoaderIF, BoxParamLoaderIF
from test.testbase import TestBase

class TestBaseLoader(TestBase):

    def test_WaveLoaderIF(self):        
        loaderIF = WaveLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        wave = loaderIF.load()

        self.assertIsNotNone(wave)
        self.assertEqual(wave.shape, (1178,))
        
    def test_FluxLoaderIF(self):
        loaderIF = FluxLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        flux = loaderIF.load()

        self.assertIsNotNone(flux)
        self.assertEqual(flux.shape, (120, 1178))
    
    def test_SpecLoaderIF(self):
        loaderIF = SpecLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        spec = loaderIF.load()

        self.assertIsNotNone(spec.flux)
        self.assertEqual(spec.flux.shape, (120, 1178))

        self.assertIsNotNone(spec.wave)
        self.assertEqual(spec.wave.shape, (1178,))

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