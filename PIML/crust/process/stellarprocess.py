from PIML.crust.data.specgrid.basespecgrid import StellarSpecGrid
from PIML.crust.operation.specgridoperation import BaseSpecGridOperation, \
    BoxSpecGridOperation, SplitSpecGridOperation, TuneSpecGridOperation, \
    LogSpecGridOperation, CoordxifySpecGridOperation, InterpSpecGridOperation, \
    SimulateSkySpecOperation, MapSNRSpecGridOperation
from PIML.crust.operation.specoperation import AddLowResObsSpecOperation
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
            # generate sky_grid with waveH from Sky file by integration
            # add self.sky
            SimulateSkySpecOperation(DATA["Sky"]),
            # split into arm 
            # modify wave, flux, (sky)
            SplitSpecGridOperation(PARAMS["arm"]),
            # map noise level to snr
            # add self.map_snr, self.map_snr_inv
            MapSNRSpecGridOperation(),

            if hasattr(PARAMS, "step") and PARAMS["step"] > 1:
                # downsample to overcome redshift
                # add self.skyH, self.step
                # modify wave, flux, sky
                TuneSpecGridOperation(MODEL_TYPES["Resolution"], PARAMS["step"]),
            else:
                PARAMS["step"] = 1
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

