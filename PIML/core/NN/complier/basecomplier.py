
from abc import ABC, abstractmethod
from tensorflow import keras

class BaseCompiler(ABC):
    @abstractmethod
    def complie(self):
        pass

class Compiler(BaseCompiler):
    def __init__(self, loss, metrics, opt, lr):
        self.loss = loss
        self.optimzer = self.create_optimzer(opt, lr)
        self.metrics = metrics

    def create_optimzer(self, name, lr):
        if name == 'adam':
            return keras.optimizers.Adam(learning_rate=lr, decay=1e-6)
        if name == 'sgd':
            return keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        else:
            raise ValueError("optimizer_type must be 'adam' or 'sgd'.")

    def compile(self, model):
        model.compile(  loss     =self.loss, 
                        metrics  =self.metrics,
                        optimizer=self.optimzer)