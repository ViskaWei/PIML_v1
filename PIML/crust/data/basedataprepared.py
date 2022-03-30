
import logging
from abc import ABC, abstractmethod

class basePrepared(ABC):
    @abstractmethod
    def print(self):
        pass

    @abstractmethod
    def get_prepared_data(self):
        pass


class WavePrepared(basePrepared):
    """ class for Wave Object """
    def __init__(self, wOrg, wPrepared, wRng, wInRng, wRngIdx):
        self.wOrg = wOrg
        self.wave = wPrepared
        self.wRng = wRng
        self.wInRng = wInRng
        self.wRngIdx = wRngIdx


    def print(self):
        logging.info("=================WavePrepared: =================")
        logging.info(f"waveOrg:  {len(self.waveOrg)}, start {self.waveOrg[0]} end {self.waveOrg[-1]}")
        logging.info(f"wave:  {len(self.wave)}, start {self.wave[0]} end {self.wave[-1]}")
        logging.info(f"wRng: {self.wRng} & waveIdx {self.wRngIdx}")

    def get_prepared_data(self):
        return self.wave