from abc import ABC, abstractmethod
from PIML.gateway.loaderIF.baseloaderIF import PathLoaderIF
from PIML.crust.process.baseprocess import BaseProcess



class BaseProcessIF(ABC):
    """ Base class for Process interface for data. """
    @abstractmethod
    def set_data(self, DATA_PARAMS):
        pass
    @abstractmethod
    def set_param(self, OP_PARAMS):
        pass
    @abstractmethod
    def set_model(self, MODEL_TYPES):
        pass
    @abstractmethod
    def interact(self, param, data):
        pass


class ProcessIF(BaseProcessIF):
    def __init__(self) -> None:
        self.loader: PathLoaderIF = None
        self.Process: BaseProcess = None

    def set_object(self, OBJECT_PARAMS):
        self.DATA_DIR = OBJECT_PARAMS["DATA_DIR"]
        self.loader.set_path(self.DATA_DIR)
        self.Object = self.loader.load()

    def setup(self, PARAMS):
        self.set_data(PARAMS["data"])
        self.set_param(PARAMS["op"])
        self.set_model(PARAMS["model"])

    def interact(self, PARAMS):
        self.set_object(PARAMS["object"])
        self.setup(PARAMS)
        self.Process.set_process(self.OP_PARAMS, self.OP_MODELS, self.OP_DATA)
        self.Process.start(self.Object)
