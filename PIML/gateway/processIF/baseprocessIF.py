import numpy as np
from abc import ABC, abstractmethod
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF
from PIML.crust.operation.baseoperation import BaseOperation, SelectOperation, SplitOperation, BoxOperation

class BaseProcessIF(ABC):
    """ Base class for Process interface for data. """
    @abstractmethod
    def interact(self, param, data):
        pass
