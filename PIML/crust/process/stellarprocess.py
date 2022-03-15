import logging
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.operation.specgridoperation import BaseSpecGridOperation, \
    BoxSpecGridOperation, SplitSpecGridOperation, TuneSpecGridOperation, \
    LogSpecGridOperation, CoordxifySpecGridOperation, InterpSpecGridOperation, \
    SimulateSkySpecOperation
from PIML.crust.operation.specoperation import LogSpecOperation, SimulateSkySpecOperation
from PIML.crust.process.baseprocess import BaseProcess


class StellarProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operationList: list[BaseSpecGridOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES, DATA):
        self.operationList = [
            BoxSpecGridOperation(PARAMS["box_name"]),
            SimulateSkySpecOperation(DATA["Sky"]),
            SplitSpecGridOperation(PARAMS["arm"]),
            SimulateObsSpecGridOperation(),
            TuneSpecGridOperation(MODEL_TYPES["Resolution"], PARAMS["step"]),
            LogSpecGridOperation(),
            CoordxifySpecGridOperation(),
            InterpSpecGridOperation(MODEL_TYPES["Interp"]),
        ]
        

    def start(self, SpecGrid: StellarSpecGrid):
        for operation in self.operationList:
            operation.perform_on_SpecGrid(SpecGrid)

