

import os
from unittest import TestCase
from PIML.gateway.loaderIF.nnloaderIF.stellarnnloaderIF import StellarNNLoaderIF

class TestStellarNNLoaderIF(TestCase):
    def test_StellarNNLoaderIF(self):
        loader = StellarNNLoaderIF()
        PARAM = {
            "path": "test/testdata/teststellarnndata/",
            "train_name": "RedM_R1000_N5_train.h5",
            "test_name" : "RedM_R1000_N5_test.h5"
        }
        loader.set_param(PARAM)
        NN = loader.load()

        self.assertEqual(NN.x_train.shape, (5, 220))
        self.assertEqual(NN.s_train.shape, (5, 220))
        self.assertEqual(NN.y_train.shape, (5, 5))

        self.assertEqual(NN.x_test.shape, (5, 220))
        self.assertEqual(NN.s_test.shape, (5, 220))
        self.assertEqual(NN.y_test.shape, (5, 5))

        self.assertTrue(NN.y_train.max() <= 1)
        self.assertTrue(NN.y_test .max() <= 1)

        self.assertTrue(NN.y_train.min() >= 0)
        self.assertTrue(NN.y_test .min() >= 0)

        self.assertTrue(NN.s_train.min() >= 0)
        self.assertTrue(NN.s_test .min() >= 0)