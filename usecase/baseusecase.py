

from abc import ABC
from abc import ABC, abstractmethod

class BaseUseCase(ABC):

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def get_data_path(self, params):
        pass

    @abstractmethod
    def get_data(self, params):
        pass

    def execute(self):
        pass