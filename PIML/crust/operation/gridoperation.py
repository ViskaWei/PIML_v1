from abc import abstractmethod
from PIML.crust.data.grid.basegrid import BaseGrid
from PIML.crust.operation.baseoperation import BaseOperation, BaseModelOperation

class BaseGridOperation(BaseOperation):
    @abstractmethod
    def perform(self, data):
        pass
    @abstractmethod
    def perform_on_Grid(self, Grid: BaseGrid) -> BaseGrid:
        pass

class  CoordxifyGridOperation(BaseGridOperation):
    def __init__(self, origin, tick) -> None:
        self.origin = origin
        self.tick = tick

    def get_scalers(self):
        self.scaler = lambda x: (x - self.origin) / self.tick
        self.rescaler = lambda x: x * self.tick + self.origin

    def perform(self, coord):
        self.get_scalers()
        return self.scaler(coord)
        
    def perform_on_Grid(self, Grid: BaseGrid) -> BaseGrid:
        Grid.coordx = self.perform(Grid.coord)
        Grid.coordx_rng = Grid.coordx.max(0) - Grid.coordx.min(0)
        Grid.coordx_scaler = self.scaler
        Grid.coordx_rescaler = self.rescaler

