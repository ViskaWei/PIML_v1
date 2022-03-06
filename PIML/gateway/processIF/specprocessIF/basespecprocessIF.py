from abc import ABC, abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.spec.basegrid import StellarGrid
from PIML.gateway.processIF.baseprocessIF import BaseProcessIF, TrimmableProcessIF, BoxableProcessIF, ResTunableProcessIF

class BaseSpecProcessIF(ABC):
    """ Base class for process interface for Spec object only. """

    @abstractmethod
    def process_spec(self, spec: StellarSpec):
        pass

    # @abstractmethod

class TrimmableSpecProcessIF(TrimmableProcessIF, BaseSpecProcessIF):
    """ class for trimmable dataIF in wavelength direction. i.e. wave, flux, etc. """

    def process_spec(self, spec: StellarSpec):
        wave = self.process_data(spec.wave)
        spec.set_wave(wave)

        flux = self.process_data(spec.flux)
        spec.set_flux(flux)

class BoxableSpecProcessIF(BoxableProcessIF, BaseSpecProcessIF):
    """ class for boxable data i.e flux, parameter, etc. """

    def process_spec(self, grid: StellarGrid):
        coord =  self.process_data(grid.coord)
        grid.set_coord(coord)

        coord_idx =  self.process_data(grid.coord_idx)
        grid.set_coord_idx(coord_idx)

        if grid.has_flux():
            flux = self.process_data(grid.flux)
            grid.set_flux(flux)

class ResTunableSpecProcessIF(ResTunableProcessIF, BaseSpecProcessIF):
    """ class for res tunable data i.e flux, parameter, etc. """

    def process_spec(self, spec: StellarSpec):
        wave = super().process_data(spec.wave)
        spec.set_wave(wave)

        flux = super().process_data(spec.flux)
        spec.set_flux(flux)