
from abc import ABC, abstractmethod
from PIML.core.method.interp.interpbuilder import InterpBuilder, RBFInterpBuilder
from PIML.gateway.storerIF.basestorerIF import PickleStorerIF


class InterpBuilderModel(ABC):
    @abstractmethod
    def store(self, DATA_DIR, name="interp"):
        pass

class RBFInterpBuilderModel(RBFInterpBuilder, InterpBuilderModel):
    def __init__(self, kernel="gaussian", epsilon=0.5) -> None:
        super().__init__(kernel, epsilon)

    def store(self, DATA_DIR, name="interp"):
        self.storer = PickleStorerIF()
        self.storer.set_path(DATA_DIR, name)
        self.storer.store(self.interpolator)
        