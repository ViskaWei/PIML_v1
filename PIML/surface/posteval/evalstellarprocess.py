import os
import numpy as np
import matplotlib.pyplot as plt
from PIML.crust.data.constants import Constants
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.gateway.processIF.specgridprocessIF.basespecgridprocessIF import StellarSpecGridProcessIF

class EvalStellarSpecGridProcess():
    def __init__(self, SP: StellarSpecGridProcessIF = None, PARAMS = None):
        if SP is None:
            self.create_SP(PARAMS)
        else:
            self.SP = SP    

        self.SpecGrid: StellarSpecGrid = self.SP.Object


    def create_SP(self, PARAMS):
        self.SP = StellarSpecGridProcessIF()
        self.SP.interact(PARAMS)

    def eval_interpolator(self, axis = 1):
        pmt0 = self.SpecGrid.box["mid"]
        pmt2 = np.copy(pmt0)
        pmt2[axis] += Constants.PHYTICK[axis]
        pmt1 = 0.5 * (pmt0 + pmt2)
        
        flux0 = self.SpecGrid.get_coord_logflux(pmt0)
        flux2 = self.SpecGrid.get_coord_logflux(pmt2)
        flux1 = self.SpecGrid.interpolator(pmt1)
        
        wave = self.SpecGrid.wave
        plt.plot(wave, flux0, label= pmt0, c='b')
        plt.plot(wave, flux1, label = pmt1, c='r')
        plt.plot(wave, flux2, label = pmt2, c='cyan')
        plt.legend()