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
    def __init__(self, model, x_test, y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def evals(self):
        self.eval_accuracy(self.x_test, self.y_test)

    def eval_accuracy(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        logging.info(f"Test loss: {score[0]}")
        logging.info(f"Test accuracy: {score[1]}")
