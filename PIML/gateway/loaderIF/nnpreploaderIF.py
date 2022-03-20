

import os
import numpy as np
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF, InterpLoaderIF
from PIML.crust.data.nndata.basennprep import NNPrep, StellarNNPrep

class NNPrepLoaderIF(BaseLoaderIF):
    pass


class StellarNNPrepLoaderIF(NNPrepLoaderIF):

    def set_path(self, DATA_DIR):
        self.DATA_DIR = DATA_DIR

    def load(self):
        interp = self.load_Interp(self.DATA_DIR)
        #FIXME 
        rng    = np.array([4., 5., 3., 5., 3.])
        return StellarNNPrep(rng, interp)

    def load_Interp(self, DATA_DIR):
        PATH = os.path.join(DATA_DIR, "interp.pickle")
        loader = InterpLoaderIF()
        return loader.load(PATH)
