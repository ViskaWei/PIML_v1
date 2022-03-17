from abc import ABC, abstractmethod
from PIML.crust.data.specdata.basespec import StellarSpec
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.operation.specoperation import BaseSpecOperation,\
    SplitSpecOperation, TuneSpecOperation, SimulateSkySpecOperation,\
    LogSpecOperation, MapSNRSpecOperation, AddPfsObsSpecOperation

from PIML.crust.operation.gridoperation import BaseGridOperation,\
    CoordxifyGridOperation

from PIML.crust.operation.boxoperation import BaseBoxOperation,\
    StellarBoxOperation

class BaseProcess(ABC):
    """ Base class for Process. """
    @abstractmethod
    def start(self, data):
        pass

class StellarSpecProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operation_list: list[BaseSpecOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES, DATA):
        self.operation_list = [
            SplitSpecOperation(PARAMS["arm"]),
            SimulateSkySpecOperation(DATA["Sky"]),
            MapSNRSpecOperation(),
            TuneSpecOperation(MODEL_TYPES["Resolution"], PARAMS["step"]),
            AddPfsObsSpecOperation(PARAMS["step"]),
            LogSpecOperation(),
        ]

    def start(self, Spec: StellarSpec):
        for operation in self.operation_list:
            operation.perform_on_Spec(Spec)


class StellarGridProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operation_list: list[BaseGridOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operation_list = [
            CoordxifyGridOperation(),
        ]

    def start(self, Grid: StellarGrid):
        for operation in self.operation_list:
            operation.perform_on_Grid(Grid)

class StellarBoxProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operation_list: list[BaseBoxOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operation_list = [
            StellarBoxOperation(PARAMS["box_name"]),
        ]

    def start(self, Grid: StellarGrid):
        for operation in self.operation_list:
            operation.perform_on_Box(Grid)