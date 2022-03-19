from abc import ABC, abstractmethod
from PIML.crust.data.nndata.basennprep import NNPrep
from PIML.crust.operation.nnoperation.basennprepoperation import BaseNNPrepOperation, DataGeneratorNNPrepOperation,\
    SamplerBuilderNNPrepOperation, CoordxifyNNPrepOperation
from PIML.crust.process.baseprocess import BaseProcess


class BaseNNPrepProcess(BaseProcess):
    @abstractmethod
    def prepare(self, data):
        pass

class NNPrepProcess(BaseNNPrepProcess):
    def __init__(self) -> None:
        super().__init__()
        self.operation_list: list[BaseNNPrepOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES, DATA):
        self.operation_list = [
            SamplerBuilderNNPrepOperation(),
            CoordxifyNNPrepOperation(PARAMS["rng"]),
            DataGeneratorNNPrepOperation(),
        ]

    def prepare(self, NNP: NNPrep):
        for operation in self.operation_list:
            operation.perform_on_NNPrep(NNP)