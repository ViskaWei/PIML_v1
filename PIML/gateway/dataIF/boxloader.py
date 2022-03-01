from abc import ABC, abstractmethod

class BaseBoxLoader(ABC):
    """ Base class for box loaders. """
    
    @abstractmethod
    def set_box(self, boxParams):
        self.R = boxParams["R"]


class BoxLoader():
    """  class for loading data into box. """
