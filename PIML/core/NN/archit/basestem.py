from keras import layers
from abc import ABC, abstractmethod
from PIML.core.NN.archit.baselayer import BaseLayer

class BaseStem():
    def __init__(self, input_dim, stem_dim=None) -> None:
        self.input_dim = input_dim
        self.stem_dim = stem_dim
        self.layer = BaseLayer()

    @abstractmethod
    def build(self, ):
        pass

    # def lay(self, model):
        

class DenseStem(BaseStem):
    def __init__(self, input_dim, act, dim) -> None:
        super().__init__(input_dim)
        self.stem_dim = dim
        self.act = act

    def build(self, ):
        stem = [
            layers.Dense(self.stem_dim),
            self.layer.lay_act(self.act)
        ]
        return stem

class ConvStem(BaseStem):
    def __init__(self, input_dim, act, num_filter) -> None:
        super().__init__(input_dim)
        self.num_filter = num_filter
        self.act = act

    def build(self, ):
        stem = [
            layers.Conv2D(32, kernel_size=(3, 3)),
            self.layer.lay_act(self.act),
            layers.MaxPooling2D(pool_size=(2, 2)),
        ]
        return stem