from abc import ABC, abstractmethod
import numpy as np
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.gateway.processIF.baseprocessIF import BaseProcessIF



class BaseGridProcessIF(ABC):
    @abstractmethod
    def process_grid(self, grid: StellarGrid):
        pass
