from abc import ABC, abstractmethod

class BaseProcessIF(ABC):
    """ Base class for Process interface for data. """
    @abstractmethod
    def interact(self, param, data):
        pass
