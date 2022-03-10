import os
import numpy as np
import matplotlib.pyplot as plt
from PIML.crust.data.constants import Constants
from PIML.gateway.processIF.specgridprocessIF.basespecgridprocessIF import StellarProcessIF

GRID_PATH="/datascope/subaru/user/swei20/data/pfsspec/import/stellar/grid"
DATA_PATH=os.path.join(GRID_PATH, "bosz_5000_RHB.h5")
DATA_PARAMS = {"DATA_PATH": DATA_PATH}
OP_PARAMS = {
    "box_name": "R",
    "arm": "RedM",
    "step": 10,
    # "wave_rng": Constants.ARM_RNGS[self.arm]
}

PARAMS = {
    "data": DATA_PARAMS,
    "op"  : OP_PARAMS,
}

MODEL_TYPES = {
        "Resolution": "Alex",
        "Interp": "RBF",
}

class EvalStellarProcess():
    def __init__(self):
        self.SP = StellarProcessIF()
        self.SP.interact(PARAMS, MODEL_TYPES)
        self.wave = self.SP.SpecGrid.wave

    def eval_interpolator(self, axis = 1):
        pmt0 = self.SP.SpecGrid.box["mid"]
        pmt2 = np.copy(pmt0)
        pmt2[axis] += Constants.PHYTICK[axis]
        pmt1 = 0.5 * (pmt0 + pmt2)
        
        flux0 = self.SP.SpecGrid.get_coord_logflux(pmt0)
        flux2 = self.SP.SpecGrid.get_coord_logflux(pmt2)
        flux1 = self.SP.SpecGrid.interpolator(pmt1)
        
        wave = self.wave
        plt.plot(wave, flux0, label= pmt0)
        plt.plot(wave, flux1, label = pmt1)
        plt.plot(wave, flux2, label = pmt2)
        plt.legend()