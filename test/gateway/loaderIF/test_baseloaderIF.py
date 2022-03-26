import numpy as np
import pandas as pd
from PIML.gateway.loaderIF.baseloaderIF import FileLoaderIF,\
    SpecGridLoaderIF, SpecLoaderIF, GridLoaderIF, \
    NNTestLoaderIF, SkyLoaderIF

from unittest import TestCase

DATA_PATH   = "test/testdata/testspecgriddata/bosz_5000_test.h5"
SKY_PATH    = "test/testdata/testspecgriddata/wavesky.npy"
INTERP_PATH = "test/testdata/testmethoddata/interp.pickel"

class TestBaseLoader(TestCase):

    def test_SpecLoaderIF(self):
        loaderIF = SpecLoaderIF()
        loaderIF.set_path(DATA_PATH)
        spec = loaderIF.load()

        self.assertIsNotNone(spec.wave)
        self.assertEqual(spec.wave.shape, (1178,))

        self.assertIsNotNone(spec.flux)
        self.assertEqual(spec.flux.shape, (120, 1178))

    def test_SpecGridLoaderIF(self):        
        loaderIF = SpecGridLoaderIF()
        loaderIF.set_path(DATA_PATH)
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
        loaderIF.set_path(DATA_PATH)
        grid = loaderIF.load()

        self.assertIsNotNone(grid.coord)
        self.assertIsNotNone(grid.coord_idx)
        self.assertIsNotNone(grid.PhyShort)

        grid.set_dfcoord()
        self.assertIsNotNone(grid.dfcoord)
        self.assertIsInstance(grid.dfcoord, pd.DataFrame)

        self.assertEqual(grid.coord.shape, (120,5))

    def test_SkyLoaderIF(self):
        loaderIF = SkyLoaderIF()
        Sky = loaderIF.load(SKY_PATH)
        sky_to_check = np.load(SKY_PATH)
        self.assertIsNone(np.testing.assert_array_equal(Sky.wave, sky_to_check[0]))
        self.assertIsNone(np.testing.assert_array_equal(Sky.sky, sky_to_check[1]))
        self.assertIsNotNone(Sky.wave2sky_fn)

    def test_FileLoaderIF(self):
        loaderIF = FileLoaderIF()
        # test pickle
        interp = loaderIF.load(INTERP_PATH)
        value = interp([[1.17,1.18]]).round(3)
        value_to_check = np.array([[1.191, 1.191]])
        self.assertIsNone(np.testing.assert_array_equal(value, value_to_check))
