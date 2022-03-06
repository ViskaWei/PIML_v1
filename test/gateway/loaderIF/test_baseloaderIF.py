import pandas as pd
from PIML.gateway.loaderIF.baseloaderIF import ObjectLoaderIF, SpecGridLoaderIF, SpecLoaderIF, GridLoaderIF
from test.testbase import TestBase

class TestBaseLoader(TestBase):

    def test_SpecLoaderIF(self):
        loaderIF = SpecLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        spec = loaderIF.load()

        self.assertIsNotNone(spec.wave)
        self.assertEqual(spec.wave.shape, (1178,))

        self.assertIsNotNone(spec.flux)
        self.assertEqual(spec.flux.shape, (120, 1178))

    def test_SpecGridLoaderIF(self):        
        loaderIF = SpecGridLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        specGrid = loaderIF.load()

        self.assertIsNotNone(specGrid.wave)
        self.assertEqual(specGrid.wave.shape, (1178,))

        self.assertIsNotNone(specGrid.flux)
        self.assertEqual(specGrid.flux.shape, (120, 1178))

        self.assertIsNotNone(specGrid.coord)
        self.assertEqual(specGrid.coord.shape, (120,5))

        self.assertIsNotNone(specGrid.coord_idx)
        self.assertEqual(specGrid.coord_idx.shape, (120,5))
        
    def test_GridLoaderIF(self):
        loaderIF = GridLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        grid = loaderIF.load()

        self.assertIsNotNone(grid.coord)
        self.assertIsNotNone(grid.coord_idx)
        self.assertIsNotNone(grid.PhyShort)

        grid.set_dfcoord()
        self.assertIsNotNone(grid.dfcoord)
        self.assertIsInstance(grid.dfcoord, pd.DataFrame)

        self.assertEqual(grid.coord.shape, (120,5))


    def test_BaseLoaderIF(self):
        for LoaderIF in ObjectLoaderIF.__subclasses__():
            loaderIF = LoaderIF()
            loaderIF.set_data_path(self.DATA_PATH)
            data = loaderIF.load()
            self.assertIsNotNone(data)