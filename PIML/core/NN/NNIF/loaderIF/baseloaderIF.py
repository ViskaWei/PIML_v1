from abc import ABC, abstractmethod
from PIML.core.NN.data.basedataloader import BaseDataLoader, MINSTDataLoader

class BaseLoaderIF(ABC):
    @abstractmethod
    def load(self):
        pass

class NNLoaderIF(BaseLoaderIF):
    """ class for loading NN. """
    
    def __init__(self, name: str) -> None:
        if name =="MINST":
            self.loader = MINSTDataLoader()
        else:
            raise ValueError("Unknown NN name")
    
    def load(self):
        return self.loader.load()
