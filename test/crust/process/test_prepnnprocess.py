from test.testbase import TestBase
from PIML.crust.process.prepnnprocess import StellarPrepNNProcess
# class TestPrepNNProcess(TestCase):
#     def test_prepnn_process(self):
#         pass
class TestStellarPrepNNProcess(TestBase):
    def test_StellarPrepNNProcess(self):
        PrepNN = self.get_PrepNN()
        
        SP = StellarPrepNNProcess()
        SP.set_process(self.D.PrepNN_Params, self.D.PrepNN_Model, self.D.PrepNN_Data)
        SP.start(PrepNN)
        self.check_PrepNN(PrepNN)