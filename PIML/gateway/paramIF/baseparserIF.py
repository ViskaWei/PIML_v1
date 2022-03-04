from abc import ABC, abstractmethod


class BaseParserIF(ABC):
    """ Base class for Parser. """
    
    @abstractmethod
    def set_parser(self, parser):
        pass

    @abstractmethod
    def get_arg(self):
        pass

