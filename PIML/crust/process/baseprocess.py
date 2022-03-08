import numpy as np
import logging
from abc import ABC, abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.operation.baseoperation import BaseOperation 
from PIML.crust.operation.specoperation import BaseSpecOperation, ArmSplitOperation, ResolutionOperation
class BaseProcess(ABC):
    """ Base class for Process. """
    @abstractmethod
    def start(self, data):
        pass

class StellarProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operationList: list[BaseSpecOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES):
        self.operationList = [
            ArmSplitOperation(PARAMS["arm"]),
            ResolutionOperation(MODEL_TYPES["Resolution"], PARAMS["step"])
        ]

    def start(self, data):
        for operation in self.operationList:
            data = operation.perform(data)
        return data

    def start_on_Spec(self, spec: StellarSpec):
        for operation in self.operationList:
            operation.perform_on_Spec(spec)


class RBFProcess(BaseProcess):
    """ class for radial basis function process. """
    




