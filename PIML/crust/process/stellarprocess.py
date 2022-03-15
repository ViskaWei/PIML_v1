from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.operation.specgridoperation import BaseSpecGridOperation, \
    BoxSpecGridOperation, SplitSpecGridOperation, TuneSpecGridOperation, \
    LogSpecGridOperation, CoordxifySpecGridOperation, InterpSpecGridOperation, \
    SimulateSkySpecOperation, MapSNRSpecGridOperation, AddPfsObsSpecGridOperation

from PIML.crust.process.baseprocess import BaseProcess


class StellarProcess(BaseProcess):
    """ class for spectral process. """
    def __init__(self) -> None:
        super().__init__()
        self.operationList: list[BaseSpecGridOperation] = None

    def set_process(self, PARAMS, MODEL_TYPES, DATA):
        self.operationList = [
            # add self.box = {...}
            BoxSpecGridOperation(PARAMS["box_name"]),
            # split into arm 
            # modify wave, flux, (sky)
            SplitSpecGridOperation(PARAMS["arm"]),
            # generate sky_grid (3200-11000) with waveH from Sky file by integration
            # cannot change order as self.wave (3000 - 14000) out of interp range
            # add self.sky
            SimulateSkySpecOperation(DATA["Sky"]),
            # map noise level to snr
            # add self.map_snr, self.map_snr_inv
            MapSNRSpecGridOperation(),
            # downsample to overcome redshift
            # add self.skyH, self.step
            # modify wave, flux, sky
            TuneSpecGridOperation(MODEL_TYPES["Resolution"], PARAMS["step"]),
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
            InterpSpecGridOperation(MODEL_TYPES["Interp"]),
        ]
        

    def start(self, SpecGrid: StellarSpecGrid):
        for operation in self.operationList:
            operation.perform_on_SpecGrid(SpecGrid)

