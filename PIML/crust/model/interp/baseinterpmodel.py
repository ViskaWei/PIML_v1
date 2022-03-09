import numpy as np
import logging
from abc import ABC, abstractmethod
from scipy.interpolate import RBFInterpolator
from PIML.crust.model.basemodel import BaseModel

class BaseInterpModel(BaseModel):
    @property
    @abstractmethod
    def name(self):
        return "BaseInterpModel"

    @abstractmethod
    def apply(self, data):
        pass

    @abstractmethod
    def train_interpolator(self, coord, value):
        pass

class RBFInterpModel(BaseInterpModel):
    @property
    def name(self):
        return "RBF"

    def apply(self, coord_to_be_interp, interpolator):
        return interpolator(coord_to_be_interp)
    
    def set_model_param(self, kernel="guassian", epsilon=0.5):
        self.kernel = kernel
        self.epsilon = epsilon

    def train_interpolator(self, coord, value):
        logging.info(f"Building RBF with gaussan kernel on data shape {value.shape}")
        interpolator = RBFInterpolator(coord, value, kernel=self.kernel, epsilon=self.epsilon)
        return interpolator

    def 

class PCARBFInterpModel(RBFInterpModel):
    pass
