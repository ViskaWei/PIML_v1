from abc import ABC, abstractmethod

from PIML.crust.data.nndata.stellar.basestellarnn import StellarNN



class BaseStellarNNOperation(ABC):
    @abstractmethod
    def perform_on_StellarNN(self, NN: StellarNN):
        pass