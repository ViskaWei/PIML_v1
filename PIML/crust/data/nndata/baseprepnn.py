from abc import ABC, abstractmethod
from PIML.crust.data.specgriddata.basespecgrid import BaseSpecGrid, StellarSpecGrid


class BasePrepNN(ABC):
    @abstractmethod
    def prepare(self, data):
        pass

class PrepNN(BasePrepNN):
    def __init__(self, interpolator, sky, arm, res):
        self.interpolator = interpolator
        self.sky   = sky
        self.arm   = arm
        self.res   = res
        self.name  = f"{arm}_R{res}"
        self.sampler = {}
        self.prepare()

    @classmethod
    def fromSpecGrid(cls, specgrid: BaseSpecGrid):
        return cls(specgrid.interpolator, specgrid.sky, specgrid.arm, specgrid.res)

        
    def prepare(self):
        self.train = {}
        self.test  = {}
        self.ntrain = 0
        self.ntest  = 0
        self.seed   = None

