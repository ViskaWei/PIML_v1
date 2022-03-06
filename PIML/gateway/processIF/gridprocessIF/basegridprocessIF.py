from abc import ABC, abstractmethod
import numpy as np
from PIML.crust.data.spec.basegrid import StellarGrid
from PIML.gateway.processIF.baseprocessIF import BoxableProcessIF, ResTunableProcessIF



class BaseGridProcessIF(ABC):
    @abstractmethod
    def process_grid(self, grid: StellarGrid):
        pass

class GridProcessIF(BoxableProcessIF, BaseGridProcessIF):
    """ class for boxable data i.e flux, parameter, etc. """

    def process_grid(self, grid: StellarGrid):
        coord =  self.process_data(grid.coord)
        grid.set_coord(coord)

        coord_idx =  self.process_data(grid.coord_idx)
        grid.set_coord_idx(coord_idx)

        if grid.has_flux:
            flux = self.process_data(grid.flux)
            grid.set_flux(flux)

        return grid