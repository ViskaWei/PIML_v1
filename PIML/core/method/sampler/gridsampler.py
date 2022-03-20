
from abc import ABC, abstractmethod


class BaseGridSampler(ABC):
    @abstractmethod
    def sample_grid(self, N):
        pass

class GridSampler(BaseGridSampler):
    # interpolator takes in coordx and returns value
    def __init__(self, sampler, base_interpolator):
        self.sampler = sampler
        self.interpolator = base_interpolator
        
    def sample_grid(self, N):
        sample_coordx = self.sampler(N)
        sample_value = self.interpolator(sample_coordx, scale=False)
        return sample_coordx, sample_value


class StellarGridSampler(GridSampler):
    def __init__(self, sampler, interpolator, rescaler):
        super().__init__(sampler, interpolator)
        self.rescaler = rescaler
    
    def sample_grid(self, N):
        unit_coordx = self.sampler(N)
        grid_coordx = self.rescaler(unit_coordx)
        sample_value = self.interpolator(grid_coordx)
        return unit_coordx, sample_value
        
        