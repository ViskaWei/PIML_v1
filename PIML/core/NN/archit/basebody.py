from keras import layers
from abc import ABC, abstractmethod
import numpy as np
from PIML.core.NN.archit.baseblock import Block, DenseBlock


class BaseBody(ABC):
    @abstractmethod
    def build(self, ):
        pass


class DenseBody(BaseBody):
    def __init__(self, stem_dim, head_dim, block: DenseBlock, scheme: str) -> None:
        self.stem_dim = stem_dim
        self.head_dim = head_dim
        self.Block = block
        self.scheme = scheme
        self.units = None

    def build(self):
        self.get_units()
        body = []
        for units in self.units:
            body = body + self.Block.lay(units)
        return body

    def get_units(self):
        if self.scheme =="Pyramid":
            self.units = self.pyramid_scheme()
        else:
            raise ValueError("Unknown scheme: {}".format(self.scheme))

    def pyramid_scheme(self):
        start_idx = np.floor(np.log2(self.stem_dim))
        end_idx = np.min(3, np.ceil(np.log2(self.head_dim)))
        units = []
        for idx in range(start_idx, end_idx, -1):
            units.append(2 ** idx)
        return units

class ConvBody(BaseBody):
    def build(self, ):
        
        pass