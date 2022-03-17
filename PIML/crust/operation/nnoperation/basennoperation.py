
from abc import ABC, abstractmethod
from PIML.core.NN.traineval.basetraineval import NNEval
from PIML.crust.operation.baseoperation import BaseOperation
from PIML.core.NN.complier.basecomplier import Compiler
from PIML.crust.data.nndata.basenn import NN

class BaseNNOperation(BaseOperation):
    @abstractmethod
    def perform_on_NN(self, model):
        pass

class BuildNNOperation(BaseNNOperation):
    def __init__(self, model_type, model_param) -> None:
        self.model_type = model_type
        self.model_param = model_param

    def perform_on_NN(self, model):
        model.set_model(self.model_type)
        model.set_model_param(self.model_param)



class CompileNNOperation(BaseNNOperation):
    def __init__(self, PARAM) -> None:
        self.Complier = Compiler(PARAM["loss"], 
                                PARAM["metrics"], 
                                PARAM["opt"],
                                PARAM["lr"])
    def perform(self, model):
        self.Complier.compile(model)      

    def perform_on_NN(self, NN):
        return self.perform(NN.model)  

class TrainNNOperation(BaseNNOperation):
    def __init__(self, PARAM) -> None:
        self.batch_size = PARAM["batch_size"]
        self.epochs = PARAM["epochs"]
        self.validation_split = PARAM["validation_split"]

    def perform(self, model, x_train, y_train):
        model.fit(x_train, y_train, \
                batch_size=self.batch_size, 
                epochs=self.epochs, 
                validation_split=self.validation_split)

    def perform_on_NN(self, NN: NN) -> None:
        self.perform(NN.model, NN.x_train, NN.y_train)

class EvalNNOperation(BaseNNOperation):
    def __init__(self) -> None:
        self.evaluator = NNEval()

    def perform(self, model, x_test, y_test):
        self.evaluator.set_eval_model(model)
        self.evaluator.set_eval_data(x_test, y_test)
        self.evaluator.evals()
        return self.evaluator.score

    def perform_on_NN(self, NN: NN) -> None:
        NN.score = self.perform(NN.model, NN.x_test, NN.y_test)