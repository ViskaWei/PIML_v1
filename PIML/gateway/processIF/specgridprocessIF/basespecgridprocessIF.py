import numpy as np
from abc import ABC, abstractmethod
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.gateway.processIF.specprocessIF.basespecprocessIF import BaseSpecProcessIF, TrimmableSpecProcessIF, BoxableSpecProcessIF, ResTunableSpecProcessIF


class BaseSpecGridProcessIF(ABC):
    @abstractmethod
    def process(self, specgrid: StellarSpecGrid) -> StellarSpecGrid:
        pass

class StellarSpecGridProcessIF(BaseSpecGridProcessIF):
    pass