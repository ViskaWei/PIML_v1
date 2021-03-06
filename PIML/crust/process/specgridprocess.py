from abc import abstractmethod
from PIML.crust.data.specgriddata.basespecgrid import BaseSpecGrid, StellarSpecGrid
from PIML.crust.operation.specgridoperation import BaseSpecGridOperation, \
    BoxSpecGridOperation, SplitSpecGridOperation, TuneSpecGridOperation, \
    LogSpecGridOperation, CoordxifySpecGridOperation, InterpSpecGridOperation, \
    SimulateSkySpecGridOperation, MapSNRSpecGridOperation, AddPfsObsSpecGridOperation
from PIML.crust.process.baseprocess import BaseProcess


class SpecGridProcess(BaseProcess):
    def __init__(self) -> None:
        self.operation_list: list[BaseSpecGridOperation] = None
    
    def set_process(self):
        pass

    def start(self, SpecGrid: BaseSpecGrid):
        for operation in self.operation_list:
            operation.perform_on_SpecGrid(SpecGrid)


class StellarSpecGridProcess(SpecGridProcess):
    """ class for spectral process. """
    def set_process(self, PARAMS, MODEL, DATA):
        self.operation_list = [
            # add self.box = {...}
            BoxSpecGridOperation(PARAMS["box_name"]),
            # split into arm 
            # modify wave, flux, (sky)
            SplitSpecGridOperation(PARAMS["arm"]),
            # generate sky_grid (3200-11000) with waveH from Sky file by integration
            # cannot change order as self.wave (3000 - 14000) out of interp range
            # add self.sky
            SimulateSkySpecGridOperation(DATA["Sky"]),
            # map noise level to snr
            # add self.map_snr, self.map_snr_inv
            MapSNRSpecGridOperation(),
            # downsample to overcome redshift
            # add self.skyH, self.step
            # modify wave, flux, sky
            TuneSpecGridOperation(MODEL["ResTune"]),
            # simulator of noise
            # add self.Obs = {...}
            AddPfsObsSpecGridOperation(PARAMS["step"]),               
            # taking log of flux
            # add self.logflux
            LogSpecGridOperation(),
            # generate coord for interpolation
            # add self.coordx, coordx_scaler, coordx_rescaler
            CoordxifySpecGridOperation(),
            # add self.interpolator
            InterpSpecGridOperation(MODEL["Interp"]),
        ]

    def start(self, SpecGrid: StellarSpecGrid):
        super().start(SpecGrid)
        



