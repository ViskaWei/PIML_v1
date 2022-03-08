import numpy as np
from abc import abstractmethod

from PIML.crust.data.grid.basegrid import BaseGrid, StellarGrid
from PIML.crust.data.constants import Constants

from PIML.crust.operation.baseoperation import BaseOperation


class BaseGridOperation(BaseOperation):
    @abstractmethod
    def perform_on_Grid(self, Grid: BaseGrid) -> BaseGrid:
        pass


class StellarBoxOperation(BaseGridOperation):
    def __init__(self, box_region: Constants.DRs.keys()) -> None:
        self.R = box_region
        

    def perform_on_Grid(self, Grid: StellarGrid) -> StellarGrid:
        pass


    @staticmethod
    def get_minmax_scaler_fns(PhyMin, PhyRng):
        def scaler_fn(x):
            return (x - PhyMin) / PhyRng
        def inverse_scaler_fn(x):
            return x * PhyRng + PhyMin        
        return scaler_fn, inverse_scaler_fn

