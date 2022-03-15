from abc import ABC, abstractmethod
from PIML.core.NN.archit.baselayer import BaseLayer


class BaseBlock(ABC):
    @abstractmethod
    def lay(self, ):
        pass

class Block(BaseBlock):
    def __init__(self, act, dropout=None) -> None:
        self.layer = BaseLayer()
        self.act = act
        self.dropout = dropout

    def lay(self, ):
        pass

    def decor(self, order):
        block = []
        for tag in order:
            block.append(self.decor_by_tag(tag))
        return block

    def decor_by_tag(self, tag):
        if (tag == "A") and (self.act is not None):
            return self.layer.lay_act(self.act)
        elif (tag == "D") and (self.dropout is not None):
            return self.layer.lay_dropout(self.dropout)
        elif tag == "N":
            return self.layer.lay_batchnorm()
        elif tag == "F":
            return self.function_layer
        else:
            raise ValueError("Unknown tag: {}".format(tag))

class DenseBlock(Block):
    def __init__(self, act, dropout=None, order="FDAN") -> None:
        super().__init__(act, dropout)
        self.order = order

    def lay(self, dim):
        block = [self.layer.lay_dense(dim)]
        block.append(self.decor(self.order))
        return block

    
