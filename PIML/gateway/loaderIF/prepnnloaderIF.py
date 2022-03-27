

import os
import numpy as np
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF, ObjectLoaderIF, SkyLoaderIF
from PIML.crust.data.nndata.baseprepnn import PrepNN

class PrepNNLoaderIF(BaseLoaderIF):
    def load(self) -> PrepNN:
        pass

class StellarPrepNNLoaderIF(PrepNNLoaderIF):

    def set_param(self, PARAMS):
        self.dir = PARAMS["path"]
        self.arm = PARAMS["arm"]
        self.res = PARAMS["res"]

    def load(self):
        interp = self.load_interp("interp.pickle")
        sky    = self.load_sky("sky.h5")
        return PrepNN(interp, sky, self.arm, self.res)

    def load_interp(self, filename="interp.pickle"):
        loader = ObjectLoaderIF()
        PATH = os.path.join(self.DATA_DIR, filename)
        return loader.load(PATH)

    def load_sky(self, filename="sky.h5"):
        loader = SkyLoaderIF()
        self.sky_path = os.path.join(self.dir, filename)
        sky = loader.load(self.sky_path, self.arm, self.res)
        return sky

