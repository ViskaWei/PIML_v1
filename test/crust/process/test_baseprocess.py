import numpy as np
from test.testbase import TestBase
from PIML.crust.data.spec.basespec import StellarSpec
from PIML.crust.data.grid.basegrid import StellarGrid
from PIML.crust.process.baseprocess import StellarSpecProcess, StellarGridProcess

class TestBaseProcess(TestBase):
    def test_BaseProcess(self):
        pass

    def test_StellarProcess(self):
        self.check_StellarSpecProcess()
        self.check_StellarGridProcess()

    def check_StellarSpecProcess(self):
        spec = StellarSpec(self.wave, self.flux)

        Process = StellarSpecProcess()
        Process.set_process(self.OP_PARAMS, self.OP_MODELS)
        Process.start(spec)

        self.assertIsNone(np.testing.assert_array_less(spec.wave.shape, self.wave.shape))
        self.assertIsNone(np.testing.assert_array_equal(spec.flux.shape[0], self.flux.shape[0]))
        self.assertIsNone(np.testing.assert_array_less(spec.flux.shape[1], self.flux.shape[1]))


    def check_StellarGridProcess(self):
        Grid = StellarGrid(self.para, self.pdx)

        Process = StellarGridProcess()
        Process.set_process(self.OP_PARAMS, self.OP_MODELS)
        Process.start(Grid)

        self.assertIsNotNone(Grid.box)
        



