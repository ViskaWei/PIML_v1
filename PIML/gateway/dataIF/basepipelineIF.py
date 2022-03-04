import numpy as np
from abc import ABC, abstractmethod

from PIML.gateway.dataIF.baseprocessIF import ResTunableProcessIF, TrimmableProcessIF
# from PIML.



class BasePipelineIF(ABC):

    @abstractmethod
    def set_param(self, param):
        pass

    @abstractmethod
    def build_pipeline(self):
        pass


class FluxPipelineIF(BasePipelineIF):

    def __init__(self):
        super().__init__()

    Pipelines = {
        TrimmableProcessIF(),
        ResTunableProcessIF()
    }

    for process
