from abc import ABC, abstractmethod
from PIML.gateway.loaderIF.baseloaderIF import PathLoaderIF
from PIML.crust.process.baseprocess import BaseProcess



class BaseProcessIF(ABC):
    """ Base class for Process interface for data. """
    @abstractmethod
    def set_data(self, DATA_PARAM):
        pass
    @abstractmethod
    def set_param(self, OP_PARAM):
        pass
    @abstractmethod
    def set_model(self, MODEL_TYPES):
        pass
    @abstractmethod
    def interact(self, param, data):
        pass


class ProcessIF(BaseProcessIF):
    def __init__(self) -> None:
        self.OP_PARAM: dict = {}
        self.OP_MODEL: dict = {}
        self.OP_DATA: dict = {}
        
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
        self.interact_on_Object(self.Object)

    def interact_on_Object(self, Object):
        self.Process.set_process(self.OP_PARAM, self.OP_MODEL, self.OP_DATA)
        self.Process.start(Object)