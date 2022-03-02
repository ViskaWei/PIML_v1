import numpy as np
from PIML.gateway.dataIF.baseloaderIF import BaseLoaderIF, WaveLoaderIF, FluxLoaderIF
from PIML.surface.database.baseloader import BaseLoader, H5pyLoader
from test.testbase import TestBase



class TestBaseLoader(TestBase):

    def test_WaveLoaderIF(self):        
        loaderIF = WaveLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        loaderIF.load_data()

        wave = loaderIF.data
        self.assertIsNotNone(wave)
        self.assertEqual(wave.shape, (1178,))
        
    def test_FluxLoaderIF(self):
        loaderIF = FluxLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        loaderIF.load_data()

        flux = loaderIF.data
        self.assertIsNotNone(flux)
        self.assertEqual(flux.shape, (120, 1178))

    def test_BaseLoaderIF(self):
        for LoaderIF in BaseLoaderIF.__subclasses__():
            loaderIF = LoaderIF()
            loaderIF.set_data_path(self.DATA_PATH)
            loaderIF.load_data()
            self.assertIsNotNone(loaderIF.data)