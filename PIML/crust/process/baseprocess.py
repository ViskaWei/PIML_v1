import numpy as np
import logging
from abc import ABC, abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.operation.baseoperation import BaseOperation, SplitOperation, ResOperation

class BaseProcess(ABC):
    """ Base class for Process. """
    @abstractmethod
    def start(self, data):
        pass

class StellarProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.ProcessList: list[BaseOperation] = []

    def set_process(self, param, model_type):
        self.ProcessList = [
            SplitOperation(param["startIdx"], param["endIdx"]),
            ResOperation(model_type["Resolution"], param["step"])
        ]

    def process_data(self, data):
        for operation in self.ProcessList:
            data = operation.perform(data)
        return data

    def process_spec(self, spec: StellarSpec):
        for operation in self.ProcessList:
            spec.wave = operation.perform(spec.wave)
            spec.flux = operation.perform(spec.flux)




# class SpecProcess(BaseProcess):
#     """ class for spectral process. """
    
#     def set_operation_with_param(self, param):
#         self.operation_pool = [
#             SplitOperation(),

#         ]





