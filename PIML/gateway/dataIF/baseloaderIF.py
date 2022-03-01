from abc import ABC, abstractmethod

from PIML.surface.database.baseloader import BaseLoader

class BaseLoaderIF(ABC):
    """ Base class for dataIF. """
    def __init__(self, ) -> None:
        super().__init__()
        self.loader = None
        self.path = None

    def set_data_path(self, DATA_PATH: str):
        self.path = DATA_PATH

    def set_loader(self, loader: BaseLoader):
        self.loader = loader    

    def load_arg(self, arg):
        return self.loader.load_arg(self.path, arg)

    def load_DArgs(self):
        return self.loader.load_DArgs(self.path)



