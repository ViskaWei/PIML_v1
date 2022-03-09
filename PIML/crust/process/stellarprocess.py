import logging
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.operation.specgridoperation import BaseSpecGridOperation, SplitSpecGridOperation, TuneSpecGridOperation, BoxSpecGridOperation
from PIML.crust.process.baseprocess import BaseProcess


class StellarProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operationList: list[BaseSpecGridOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operationList = [
            BoxSpecGridOperation(PARAMS["box_name"]),
            SplitSpecGridOperation(PARAMS["arm"]),
            TuneSpecGridOperation(MODEL_TYPES["Resolution"], PARAMS["step"])
        ]

    def start(self, SpecGrid: StellarSpecGrid):
        for operation in self.operationList:
            operation.perform_on_SpecGrid(SpecGrid)
