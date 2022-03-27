


from PIML.crust.process.baseprocess import BaseProcess
from PIML.crust.operation.nnoperation.basennoperation import BaseNNOperation, \
    BuildNNOperation, CompileNNOperation, EvalNNOperation, \
    TrainNNOperation

class NNProcess(BaseProcess):
    def __init__(self) -> None:
        self.operation_list: list[BaseNNOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operation_list = [
            BuildNNOperation(MODEL_TYPES["Model"], PARAMS["Model"]),
            CompileNNOperation(PARAMS["Compile"]),
            TrainNNOperation(PARAMS["Train"]),
            EvalNNOperation(PARAMS["Eval"]),
        ]
        

    def start(self, NN):
        for operation in self.operation_list:
            operation.perform_on_NN(NN)
