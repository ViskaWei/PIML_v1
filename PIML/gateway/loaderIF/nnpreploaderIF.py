

import os
import numpy as np
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF, FileLoaderIF
from PIML.crust.data.nndata.basennprep import NNPrep

class NNPrepLoaderIF(BaseLoaderIF):
    def load(self) -> NNPrep:
        pass

class StellarNNPrepLoaderIF(NNPrepLoaderIF):
    def set_path(self, DATA_DIR):
        self.DATA_DIR = DATA_DIR

    def load(self):
        interp = self.load_file("interp.pickle")
        sky    = self.load_file("sky.npy")
        return NNPrep(interp, sky)

    def load_file(self, filename):
        loader = FileLoaderIF()
        PATH = os.path.join(self.DATA_DIR, filename)
        return loader.load(PATH)