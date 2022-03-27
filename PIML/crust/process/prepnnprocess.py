from abc import ABC, abstractmethod
from PIML.crust.data.nndata.baseprepnn import PrepNN
from PIML.crust.operation.nnoperation.baseprepnnoperation import BasePrepNNOperation,\
    DataGeneratorPrepNNOperation, UniformLabelSamplerPrepNNOperation,\
    HaltonLabelSamplerPrepNNOperation, CoordxifyPrepNNOperation,\
    AddPfsObsNNPredOperation, TrainPrepNNOperation, TestPrepNNOperation
from PIML.crust.process.baseprocess import BaseProcess


class BasePrepNNProcess(BaseProcess):
    # set_process, start
    pass

class PrepNNProcess(BasePrepNNProcess):
    pass

class StellarPrepNNProcess(PrepNNProcess):
    def __init__(self) -> None:
        self.operation_list: list[BasePrepNNOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES, DATA):
        self.operation_list = [
            CoordxifyPrepNNOperation(DATA["rng"]),
            AddPfsObsNNPredOperation(PARAMS["step"]),
            UniformLabelSamplerPrepNNOperation(),
            HaltonLabelSamplerPrepNNOperation(),
            DataGeneratorPrepNNOperation(),
            TrainPrepNNOperation(PARAMS["ntrain"]),
            TestPrepNNOperation(PARAMS["ntest"]),

        ]

    def start(self, NNP: PrepNN):
        for operation in self.operation_list:
            operation.perform_on_PrepNN(NNP)
                

    