from abc import ABC, abstractmethod
from PIML.crust.data.constants import Constants
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.operation.baseoperation import BaseOperation, BaseModelOperation
from PIML.crust.operation.specoperation import SplitSpecOperation, TuneSpecOperation
from PIML.crust.operation.gridoperation import CoordxifyGridOperation
from PIML.crust.operation.boxoperation import StellarBoxOperation

from PIML.crust.model.interp.baseinterpmodel import BaseInterpModel, RBFInterpModel, PCARBFInterpModel

class BaseSpecGridOperation(BaseOperation):
    @abstractmethod
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        pass

class CoordxifySpecGridOperation(BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        if isinstance("box", SpecGrid):
            origin = SpecGrid.box["min"]
        else:
            # raise NotImplementedError
            origin = SpecGrid.coord.min(0)

        OP = CoordxifyGridOperation(origin, Constants.PHYTICK)
        OP.perform_on_Grid(SpecGrid)


class InterpSpecGridOperation(BaseModelOperation, BaseSpecGridOperation):
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

    def perform(self, data):
        return self.model.apply(data)

    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> StellarSpecGrid:
        OP = CoordxifySpecGridOperation()
        OP.perform_on_SpecGrid(SpecGrid)
        self.model.train_interpolator(SpecGrid)







class BoxSpecGridOperation(StellarBoxOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> StellarSpecGrid:
        self.perform_on_Box(SpecGrid)

class SplitSpecGridOperation(SplitSpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> StellarSpecGrid:
        self.perform_on_Spec(SpecGrid)

class TuneSpecGridOperation(TuneSpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> StellarSpecGrid:
        self.perform_on_Spec(SpecGrid)
