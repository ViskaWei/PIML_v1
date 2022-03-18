from abc import ABC, abstractmethod
from PIML.crust.data.specgriddata.basespecgrid import StellarSpecGrid
from PIML.crust.data.nndata.basennprep import BaseNNPrep, NNPrep, SpecGridNNPrep
from PIML.crust.operation.baseoperation import BaseOperation

class BaseNNPrepOperation(BaseOperation):
    """ Base class for Process of preparing NN data. """
    @abstractmethod
    def perform(self, SP:StellarSpecGrid):
        pass

class SpecGridNNPrepOperation(BaseNNPrepOperation):
    def __init__(self, SP: StellarSpecGrid) -> None:
        self.SP = SP
    
    def perform(self, odx, num_ftr):
        coordx = self.SP.coordx[:, odx]
        



    def perform_on_NNPrep(self, NNP: SpecGridNNPrep): 


