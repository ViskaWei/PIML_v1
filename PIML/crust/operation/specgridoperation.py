from abc import ABC, abstractmethod
from PIML.crust.data.constants import Constants
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.operation.baseoperation import BaseOperation, BaseModelOperation
from PIML.crust.operation.specoperation import SimulateSkySpecOperation,\
    SplitSpecOperation, TuneSpecOperation, LogSpecOperation, \
    MapSNRSpecOperation, AddPfsObsSpecOperation
from PIML.crust.operation.gridoperation import CoordxifyGridOperation
from PIML.crust.operation.boxoperation import StellarBoxOperation

from PIML.crust.model.specgrid.interpspecgridmodel import InterpSpecGridModel, RBFInterpSpecGridModel, PCARBFInterpSpecGridModel


class BaseSpecGridOperation(BaseOperation):
    @abstractmethod
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        pass

class CoordxifySpecGridOperation(BaseSpecGridOperation):
    def perform(self, coord):
        origin = coord.min(0)
        OP = CoordxifyGridOperation(origin, Constants.PHYTICK)
        return OP.perform(coord)

    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        if hasattr(SpecGrid, "box"):
            origin = SpecGrid.box["min"]
        else:
            origin = SpecGrid.coord.min(0)

        OP = CoordxifyGridOperation(origin, Constants.PHYTICK)
        OP.perform_on_Grid(SpecGrid)


class InterpSpecGridOperation(BaseSpecGridOperation):
    def __init__(self, model_type) -> None:
        self.model = self.set_model(model_type)
        self.model.set_model_param()
    
    def set_model(self, model_type: str) -> InterpSpecGridModel:
        if model_type == "RBF":
            model = RBFInterpSpecGridModel()
        elif model_type == "PCARBF":
            model = PCARBFInterpSpecGridModel()
        else:
            raise ValueError("Unknown Interp model type: {}".format(model_type))
        return model

    def perform(self, data):
        return self.model.apply(data)

    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.model.apply_on_SpecGrid(SpecGrid)

class BoxSpecGridOperation(StellarBoxOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.perform_on_Box(SpecGrid)

class SplitSpecGridOperation(SplitSpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.perform_on_Spec(SpecGrid)

class TuneSpecGridOperation(TuneSpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.perform_on_Spec(SpecGrid)

class LogSpecGridOperation(LogSpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.perform_on_Spec(SpecGrid)

class SimulateSkySpecGridOperation(SimulateSkySpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.perform_on_Spec(SpecGrid)

class MapSNRSpecGridOperation(MapSNRSpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.perform_on_Spec(SpecGrid)

class AddPfsObsSpecGridOperation(AddPfsObsSpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> None:
        self.perform_on_Spec(SpecGrid)