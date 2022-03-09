import numpy as np
import logging
from abc import ABC, abstractmethod
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.operation.baseoperation import BaseOperation
from PIML.crust.operation.gridoperation import BaseGridOperation 
from PIML.crust.operation.specoperation import BaseSpecOperation, SplitSpecOperation, TuneSpecOperation
from PIML.crust.operation.boxoperation import BaseBoxOperation, StellarBoxOperation

class BaseProcess(ABC):
    """ Base class for Process. """
    @abstractmethod
    def start(self, data):
        pass

class StellarSpecProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operationList: list[BaseSpecOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operationList = [
            SplitSpecOperation(PARAMS["arm"]),
            TuneSpecOperation(MODEL_TYPES["Resolution"], PARAMS["step"])
        ]

    def start(self, Spec: StellarSpec):
        for operation in self.operationList:
            if isinstance(operation, BaseSpecOperation):
                operation.perform_on_Spec(Spec)


class StellarGridProcess(BaseProcess):

    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operationList: list[BaseSpecOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operationList = [
            StellarBoxOperation(PARAMS["box_name"]),
        ]

    def start(self, Grid: StellarGrid):
        for operation in self.operationList:
            if isinstance(operation, BaseBoxOperation):
                operation.perform_on_Box(Grid)
            elif isinstance(operation, BaseGridOperation):
                operation.perform_on_Grid(Grid)