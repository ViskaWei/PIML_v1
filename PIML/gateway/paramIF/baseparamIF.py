from abc import ABC, abstractmethod
from PIML.surface.parser.baseparser import BaseParser

class BaseParamIF(ABC):

    @abstractmethod
    def set_data(self, data):
        pass

    # @abstractmethod


class TrimmableParamIF(BaseParamIF):
    
    @abstractmethod
    def set_data(self, data):
        pass