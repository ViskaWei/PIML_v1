import numpy as np
import logging
from abc import ABC, abstractmethod

class BaseProcess(ABC):
    """ Base class for Process. """

    @abstractmethod
    def set_process_param(self, param):
        pass

    @abstractmethod
    def set_process_data(self, data):
        pass

    @abstractmethod
    def process(self, data):
        pass

    @abstractmethod
    def print(self):
        pass

