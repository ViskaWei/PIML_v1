from abc import abstractmethod
import numpy as np
from PIML.crust.data.spec.baseobs import StellarObs
from PIML.crust.model.spec.basespecmodel import BaseSpecModel
from PIML.crust.data.spec.basespec import StellarSpec

class ObsModel(BaseSpecModel):
    @abstractmethod
    def simulate_sky(self, data):
        pass

class PfsObsModel(ObsModel):
    def __init__(self, Obs: StellarObs) -> None:
        self.Obs = Obs

    @property
    def name(self):
        return "Pfs"

    def apply(self, wave, flux=None):
        sky = self.Obs.rebin_sky(wave)
        if flux is not None:
            
        return sky
        


    def apply_on_Spec(self, Spec: StellarSpec) -> StellarSpec:
        self.apply(Spec.wave)        

