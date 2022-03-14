from abc import ABC, abstractmethod
import logging

class BaseEval(ABC):
    @abstractmethod
    def eval_accuracy(self):
        pass
    @abstractmethod
    def evals(self):
        pass

class NNEval(BaseEval):
    def __init__(self, *args, **kwargs):
        self.model = None
        self.score = None
    
    def set_eval_data(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test

    def set_eval_model(self, model):
        self.model = model

    def evals(self):
        self.eval_accuracy(self.x_test, self.y_test)

    def eval_accuracy(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        logging.info(f"Test loss: {score[0]}")
        logging.info(f"Test accuracy: {score[1]}")
        self.score = score
