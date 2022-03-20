
from PIML.gateway.loaderIF.nnpreploaderIF import StellarNNPrepLoaderIF
from PIML.gateway.processIF.baseprocessIF import BaseProcessIF


class NNPrepProcessIF(BaseProcessIF):
    def interact_on_NNPrep(self, param, data):
        pass

class StellarNNPrepProcessIF(NNPrepProcessIF):

    def __init__(self) -> None:
        self.loader = StellarNNPrepLoaderIF()   


    def set_object(self, OBJECT_PARAMS):
        self.Object = loader.load(OBJECT_PARAMS["DATA_PATH"])

