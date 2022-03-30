from abc import ABC, abstractmethod
from PIML.crust.data.nndata.baseprepnn import PrepNN
from PIML.crust.operation.nnoperation.baseprepnnoperation import BasePrepNNOperation,\
    UniformLabelSamplerPrepNNOperation,\
    HaltonLabelSamplerPrepNNOperation, CoordxifyPrepNNOperation,\
    AddPfsObsNNPredOperation, LabelPrepNNOperation, DataPrepNNOperation,\
    FinishPrepNNOperation

from PIML.crust.process.baseprocess import BaseProcess


class BasePrepNNProcess(BaseProcess):
    # set_process, start
    pass

class PrepNNProcess(BasePrepNNProcess):
    pass

class StellarPrepNNProcess(PrepNNProcess):
    def __init__(self) -> None:
        self.operation_list: list[BasePrepNNOperation] = None

    def set_process(self, PARAMS, MODEL, DATA):
        self.operation_list = [
            CoordxifyPrepNNOperation(DATA["rng"]),
            AddPfsObsNNPredOperation(PARAMS["step"]),
            UniformLabelSamplerPrepNNOperation(),
            HaltonLabelSamplerPrepNNOperation(),
            # DataGeneratorPrepNNOperation(),
            LabelPrepNNOperation(PARAMS["ntrain"],PARAMS["ntest"], PARAMS["seed"]),
            DataPrepNNOperation(),
            FinishPrepNNOperation(),

        ]

    def start(self, NNP: PrepNN):
        for operation in self.operation_list:
            operation.perform_on_PrepNN(NNP)
                

    