from abc import abstractmethod
from PIML.crust.data.specdata.basespec import StellarSpec
from PIML.crust.process.baseprocess import StellarSpecProcess
from PIML.gateway.processIF.baseprocessIF import ProcessIF
from PIML.gateway.loaderIF.baseloaderIF import SpecLoaderIF, SkyLoaderIF


class BaseSpecProcessIF(ProcessIF):
    """ Base class for process interface for Spec object only. """
    @abstractmethod
    def interact_on_Spec(self, param, Spec: StellarSpec):
        pass

class StellarSpecProcessIF(BaseSpecProcessIF):
    def __init__(self) -> None:
        super().__init__()
        self.loader = SpecLoaderIF()
        self.Process = StellarSpecProcess()

    def set_data(self, DATA_PARAMS):
        self.SKY_PATH = DATA_PARAMS["SKY_PATH"]
        self.Sky = SkyLoaderIF().load(self.SKY_PATH)
        self.OP_DATA = {"Sky": self.Sky}

    def set_param(self, OP_PARAMS):
        self.OP_PARAMS = self.paramIF(OP_PARAMS)
    
    def set_model(self, MODEL_TYPES):
        self.OP_MODELS = MODEL_TYPES

    def interact_on_Spec(self, PARAMS, Spec: StellarSpec):
        self.setup(PARAMS)
        self.interact_on_object(Spec)

    def paramIF(self, PARAMS):
        #TODO create class later
        
        return PARAMS

