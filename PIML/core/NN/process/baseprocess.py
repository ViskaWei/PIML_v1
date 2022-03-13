from abc import ABC, abstractmethod


class BaseProcess(ABC):
    @abstractmethod
    def process(self):
        pass

class Process(BaseProcess):
    pass