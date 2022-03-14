


from PIML.crust.process.baseprocess import BaseProcess
from PIML.crust.operation.nnoperation import BaseNNOperation, CompileNNOperation, EvalNNOperation, \
TrainNNOperation

class NNProcess(BaseProcess):
    def __init__(self) -> None:
        self.operationList: list[BaseNNOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operationList = [
            CompileNNOperation(PARAMS["Compile"]),
            TrainNNOperation(PARAMS["Train"]),
            EvalNNOperation(PARAMS["Eval"]),
        ]
        

    def start(self, NN):
        for operation in self.operationList:
            operation.perform_on_NN(NN)
