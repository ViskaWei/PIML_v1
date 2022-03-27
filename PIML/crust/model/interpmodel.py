
from abc import ABC, abstractmethod
from PIML.core.method.interp.interpbuilder import InterpBuilder, RBFInterpBuilder
from PIML.gateway.storerIF.basestorerIF import InterpStorerIF


class InterpBuilderModel(ABC):
    @abstractmethod
    def store(self, DATA_DIR, name="interp"):
        pass

class RBFInterpBuilderModel(RBFInterpBuilder, InterpBuilderModel):
    def __init__(self, kernel="gaussian", epsilon=0.5) -> None:
        super().__init__(kernel, epsilon)

    def apply(self, coord, value):
        self.build(coord, value)
        def interpolator(eval_coord):
            if eval_coord.ndim == 1:
                return self.interpolator([eval_coord])[0]
            else:
                return self.interpolator(eval_coord)
        return interpolator

    def store(self, DATA_DIR, name="interp"):
        self.storer = InterpStorerIF()
        self.storer.set_dir(DATA_DIR, name)
        self.storer.store(self.interpolator)
        