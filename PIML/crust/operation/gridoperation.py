import numpy as np
from abc import abstractmethod

from PIML.crust.data.grid.basegrid import BaseGrid, StellarGrid
from PIML.crust.data.constants import Constants
from PIML.crust.model.interp.baseinterpmodel import BaseInterpModel, RBFInterpModel, PCARBFInterpModel

from PIML.crust.operation.baseoperation import BaseOperation, BaseModelOperation


class BaseGridOperation(BaseOperation):
    @abstractmethod
    def perform_on_Grid(self, Grid: BaseGrid) -> BaseGrid:
        pass

    def perform(self,data):
        pass

    @staticmethod
    def get_coordx_scaler_fns(box_min):
        def scaler_fn(x):
            return np.divide((x - box_min) ,Constants.PhyTick)
        def inverse_scaler_fn(x):
            return x * Constants.PhyTick + box_min
        return scaler_fn, inverse_scaler_fn


class InterpGridOperation(BaseModelOperation, BaseGridOperation):

    def __init__(self, model_type, model_param) -> None:
        super().__init__(model_type, model_param)

    def set_model(self, model_type) -> BaseInterpModel:
        if model_type == "RBF":
            model = RBFInterpModel()
        elif model_type == "PCARBF":
            model = PCARBFInterpModel()
        else:
            raise ValueError("Unknown Interp model type: {}".format(model_type))
        return model

    def perform(self, coord_idx):
        pass
        

    
    def perform_on_Grid(self, Grid: BaseGrid) -> BaseGrid:
        pass
        #
        # coord_to_interp = Grid.coord_idx
        # if (idx[0] == 0).all():
        # coord_to_interp = 

