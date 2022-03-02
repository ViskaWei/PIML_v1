from test.testbase import TestBase
from PIML.gateway.dataIF.baseprepareIF import BasePrepareIF, WavePrepareIF
from PIML.gateway.dataIF.baseloaderIF import WaveLoaderIF, FluxLoaderIF


class TestBaseDataIF(TestBase):

    
    def test_WaveDataIF(self):
        

        loaderIF = WaveLoaderIF()
        loaderIF.set_data_path(self.DATA_PATH)
        loaderIF.load_data()

        prepareIF = WavePrepareIF()
        prepareIF.prepare(loaderIF.param, loaderIF.data)



    def test_BaseDataIF(self):
        pass
