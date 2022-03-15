from PIML.crust.data.spec.obs.baseobs import Obs



class StellarObs(Obs):
    def __init__(self, ) -> None:
        pass




    def simulate(self, Spec: StellarSpec) -> StellarSpec:
        obsfluxs = self.Obs.simulate(flux)
        sigma = self.ObsSNR.convert(obsfluxs)
        return self.Obs.simulate(obsfluxs, sigma)