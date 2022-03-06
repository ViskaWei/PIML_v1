import numpy as np
import logging
from abc import ABC, abstractmethod
from PIML.crust.operation.baseoperation import BaseOperation, SelectOperation, SplitOperation, BoxOperation

class BaseProcess(ABC):
    """ Base class for Process. """
    @abstractmethod
    def start(self, data):
        pass

class SpecProcess(BaseProcess):
    """ class for spectral process. """
    
    def set_operation_with_param(self, param):
        self.operation_pool = [
            SplitOperation(),

        ]





