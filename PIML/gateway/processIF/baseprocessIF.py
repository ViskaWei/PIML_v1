from abc import ABC, abstractmethod
from PIML.crust.process.baseprocess import BaseProcess
from PIML.gateway.loaderIF.baseloaderIF import ParamLoaderIF
from PIML.gateway.storerIF.basestorerIF import BaseStorerIF

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
    def set_out(self, out):
        pass
    @abstractmethod
    def interact(self, param, data):
        pass

class ProcessIF(BaseProcessIF):
    def __init__(self) -> None:
        self.OP_PARAM: dict = {}
        self.OP_MODEL: dict = {}
        self.OP_DATA : dict = {}
        self.OP_OUT  : dict = {}
        
        self.loader : ParamLoaderIF = None
        self.Process: BaseProcess = None
        self.storer : BaseStorerIF = None

    def set_object(self, OBJECT_PARAMS):
        self.loader.set_param(OBJECT_PARAMS)
        self.Object = self.loader.load()

    def setup(self, PARAMS):
        self.set_data(PARAMS["data"])
        self.set_param(PARAMS["op"])
        self.set_model(PARAMS["model"])
        if "out" in PARAMS:
            self.set_out(PARAMS["out"])

    def interact(self, PARAMS):
        self.set_object(PARAMS["object"])
        self.setup(PARAMS)
        self.interact_on_Object(self.Object)

    def interact_on_Object(self, Object):
        self.Process.set_process(self.OP_PARAM, self.OP_MODEL, self.OP_DATA)
        self.Process.start(Object)
    

    def set_out(self, PARAMS):
        self.OP_OUT = PARAMS
        

    def finish(self, store_path):
        # store
        pass