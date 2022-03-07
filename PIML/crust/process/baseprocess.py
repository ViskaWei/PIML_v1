import numpy as np
import logging
from abc import ABC, abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.operation.baseoperation import BaseOperation, BaseSpecOperation, SplitOperation, ResolutionOperation

class BaseProcess(ABC):
    """ Base class for Process. """
    @abstractmethod
    def start(self, data):
        pass

class StellarProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operationList: list[BaseOperation] = []

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operationList = [
            SplitOperation(PARAMS["split_idxs"]),
            ResolutionOperation(MODEL_TYPES["Resolution"], PARAMS["step"])
        ]

    def start(self, data):
        for operation in self.operationList:
            data = operation.perform(data)
        return data

    def start_on_Spec(self, spec: StellarSpec):
        for operation in self.operationList:
            if operation in BaseSpecOperation.__subclasses__():
                spec = operation.perform_on_spec(spec)
            else:
                spec.wave = operation.perform(spec.wave)
                spec.flux = operation.perform(spec.flux)






