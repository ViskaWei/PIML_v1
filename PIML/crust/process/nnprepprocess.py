from abc import ABC, abstractmethod
from PIML.crust.process.baseprocess import BaseProcess

class BaseNNPrepProcess(BaseProcess):
    @abstractmethod
    def prepare(self, data):
        pass

class NNPrepProcess(BaseNNPrepProcess):
    def __init__(self) -> None:
        super().__init__()
        # self.operation_list: list[BaseNNPrepOperation] = None

    # def set_process(self, PARAMS, MODEL_TYPES):
    #     self.operation_list = [
    #         BuildNNOperation(MODEL_TYPES["Model"], PARAMS["Model"]),
    #         CompileNNOperation(PARAMS["Compile"]),
    #         TrainNNOperation(PARAMS["Train"]),
    #         EvalNNOperation(PARAMS["Eval"]),
    #     ]

    # def prepare(self, data):
    #     for operation in self.operation_list:
    #         operation.perform_on_data(data)