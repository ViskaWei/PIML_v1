from abc import abstractmethod
from PIML.crust.data.grid.basegrid import BaseGrid
from PIML.crust.operation.baseoperation import BaseOperation, BaseModelOperation, CoordxifyOperation

class BaseGridOperation(BaseOperation):
    @abstractmethod
    def perform(self, data):
        pass
    @abstractmethod
    def perform_on_Grid(self, Grid: BaseGrid) -> BaseGrid:
        pass

class  CoordxifyGridOperation(CoordxifyOperation, BaseGridOperation):
    def perform_on_Grid(self, Grid: BaseGrid) -> BaseGrid:
        Grid.coordx = self.perform(Grid.coord)
        Grid.coordx_rng = Grid.coordx.max(0) - Grid.coordx.min(0)
        Grid.coordx_scaler = self.scaler
        Grid.coordx_rescaler = self.rescaler

