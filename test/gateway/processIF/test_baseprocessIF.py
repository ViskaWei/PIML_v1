import numpy as np
from test.testbase import TestBase
from PIML.gateway.processIF.baseprocessIF import BaseModelProcessIF, BaseParamProcessIF, TrimmableProcessIF

# from PIML.gateway.dataIF.baseloaderIF import WaveLoaderIF, FluxLoaderIF


class TestBaseProcessIF(TestBase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.data1D = np.arange(10)
        self.data2D = np.tile(self.data1D, (10,1))
        

    def test_TrimmableProcessIF(self):
        startIdx, endIdx = np.digitize([4,6], self.data1D)
        param = {"startIdx": startIdx, "endIdx": endIdx}

        trimmable = TrimmableProcessIF()
        trimmable.set_param(param)
        trimmable.set_data(self.data1D)

        # testing on 1D data
        dataProcessed = trimmable.process()
        np.testing.assert_array_equal(dataProcessed, self.data1D[startIdx:endIdx])        

        # testing on 2D data
        trimmable.set_data(self.data2D)
        dataProcessed = trimmable.process()
        np.testing.assert_array_equal(dataProcessed, self.data2D[:, startIdx:endIdx])        

    def test_BoxableProcessIF(self):
        param = {"IdxInBox": [9,2,2,1,7,8]}





    def test_BaseProcessIF(self):
        param = {
            "startIdx": 3,
            "endIdx": 8,
            "IdxInBox": [9,2,2,1,7,8],
            "step": 10,
        }

        METHOD = {
            "ResTunable_model_type": "Alex"
        }

        for ProcessIF in BaseParamProcessIF.__subclasses__():
            processIF = ProcessIF()

            if ProcessIF in BaseModelProcessIF.__subclasses__():
                processIF.set_model_type(METHOD)

            processIF.set_param(param)
            processIF.set_data(self.flux)
            dataProcessed = processIF.process()
            self.assertIsNotNone(dataProcessed)

