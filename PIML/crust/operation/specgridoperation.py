from abc import ABC, abstractmethod
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.operation.boxoperation import StellarBoxOperation
from .baseoperation import BaseOperation
from PIML.crust.operation.specoperation import SplitSpecOperation, TuneSpecOperation


class BaseSpecGridOperation(BaseOperation):
    @abstractmethod
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid):
        pass

class BoxSpecGridOperation(StellarBoxOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> StellarSpecGrid:
        self.perform_on_Box(SpecGrid)

class SplitSpecGridOperation(SplitSpecOperation, BaseSpecGridOperation):
    def __init__(self, arm: str,) -> None:
        super().__init__(arm)
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> StellarSpecGrid:
        self.perform_on_Spec(SpecGrid)

class TuneSpecGridOperation(TuneSpecOperation, BaseSpecGridOperation):
    def perform_on_SpecGrid(self, SpecGrid: StellarSpecGrid) -> StellarSpecGrid:
        self.perform_on_Spec(SpecGrid)

