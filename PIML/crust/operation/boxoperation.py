import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.data.constants import Constants
from PIML.crust.data.grid.basegrid import BaseGrid, StellarGrid
from PIML.crust.operation.baseoperation import BaseOperation

class BaseBoxOperation(BaseOperation):
    def __init__(self, box_name) -> None:
        self.name = box_name
        self.box = {"name": box_name}

    @abstractmethod
    def perform_on_Box(self, Box: BaseGrid) -> BaseGrid:
        pass

class StellarBoxOperation(BaseBoxOperation):
    def __init__(self, box_name: Constants.DRs.keys()) -> None:
        super().__init__(box_name)

    def perform(self, dfcoord=None):
        self.set_box_bnd()
        if dfcoord is not None: 
            self.is_coord_inbox(dfcoord)
        return self.box

    def perform_on_Box(self, Box: StellarGrid) -> StellarGrid:
        _=self.perform(Box.dfcoord)
        Box.box = self.box

    def set_box_bnd(self):
        box_min, box_max, box_rng, box_num, box_mid = self.get_box_bnd(self.name)
        self.box["min"] = box_min
        self.box["max"] = box_max
        self.box["rng"] = box_rng
        self.box["num"] = box_num
        self.box["mid"] = box_mid

    def get_box_bnd(self, box_name):
        bnd = np.array(Constants.DRs[box_name])
        box_min, box_max = bnd.T
        box_rng = np.diff(bnd).T[0]
        box_num = box_rng / Constants.PHYTICK 
        box_mid = (box_num //2) * Constants.PHYTICK + box_min
        return box_min, box_max, box_rng, box_num, box_mid
    
    def is_coord_inbox(self, dfcoord):
        assert (dfcoord.min().values <= self.box["min"]).all() 
        assert (dfcoord.max().values >= self.box["max"]).all()

    # @staticmethod
    # def get_minmax_scaler_fns(box_min, box_rng):
    #     def scaler_fn(x):
    #         return (x - box_min) / box_rng
    #     def inverse_scaler_fn(x):
    #         return x * box_rng + box_min        
    #     return scaler_fn, inverse_scaler_fn

    

# def set_box_scaler(self):
#         self.minmax_scaler, self.minmax_rescaler = BaseBoxOperation.get_minmax_scaler_fns(self.box_min, self.box_rng)
#         self.unitcoord_scaler, self.unitcoord_rescaler = BaseBoxOperation.get_unitcoord_scaler_fns(self.box_min)
