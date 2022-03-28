from test.testbase import TestBase
from PIML.crust.process.prepnnprocess import StellarPrepNNProcess
# class TestPrepNNProcess(TestCase):
#     def test_prepnn_process(self):
#         pass
class TestStellarPrepNNProcess(TestBase):
    def test_StellarPrepNNProcess(self):
        PrepNN = self.get_PrepNN()
        self.assertIsNotNone(PrepNN.sky)
        
        SP = StellarPrepNNProcess()
        SP.set_process(self.D.PREPNN_PARAM, self.D.PREPNN_MODEL, self.D.PREPNN_DATA)
        SP.start(PrepNN)
        self.check_PrepNN(PrepNN)