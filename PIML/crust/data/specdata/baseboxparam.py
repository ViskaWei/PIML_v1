import pandas as pd
from abc import ABC, abstractmethod
from PIML.crust.data.constants import Constants
class BaseBoxParam(ABC):
    required_attributes = ["para", "dfpara"]

    @abstractmethod
    def set_dfpara(self, para):
        pass

class BoxParam(BaseBoxParam):
    def __init__(self, para, pdx=None):
        self.para = para
        self.pdx  = pdx
        self.PhyShort = Constants.PhyShort

    def set_dfpara(self):
        self.dfpara = pd.DataFrame(self.para, columns=self.PhyShort)