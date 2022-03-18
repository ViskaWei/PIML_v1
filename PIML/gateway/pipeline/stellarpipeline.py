
import numpy as np
from PIML.gateway.pipeline.basepipeline import BasePipeline 
from PIML.gateway.processIF.specgridprocessIF.basespecgridprocessIF import StellarProcessIF



class StellarDataPipeline(BasePipeline):
    """ Pipeline for SpecGrid. """
    def __init__(self) -> None:
        self.pipeline = None

        

    def set_SpecGrid_pipeline(self,PARAMS):
        SP = StellarProcessIF()
        SP.interact(PARAMS["SpecGrid"])
        return SP.Object


    def build(self, ):
    #     for process in self.process_list:
    #         P = process
    #         P.interact(t.PARAMS)
    #         Ob = SP.Object
        pass
    
