from abc import ABC, abstractmethod
from dotmap import DotMap

class BaseParser(ABC):
    """ Base class for Parser. """
    @abstractmethod
    def parse_arg(self, CONFIG):
        pass

class JSONParser(BaseParser):
    """ class for JSONParser. """
    def parse_arg(self, CONFIG):
        return DotMap(CONFIG)

class DictParser(BaseParser):
    """ class for DictParser. """
    def parse_arg(self, CONFIG):
        return CONFIG
