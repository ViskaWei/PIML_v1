
from abc import ABC, abstractmethod
from PIML.crust.operation.baseoperation import BaseOperation
from PIML.core.NN.complier.basecomplier import Compiler

class BaseNNOperation(BaseOperation):
    @abstractmethod
    def perform_on_NN(self, model):
        pass

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

# class 