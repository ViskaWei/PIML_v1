
import numpy as np
import logging
from scipy.interpolate import RBFInterpolator
from .baseinterpmodel import BaseInterpModel

class RBF(BaseInterpModel):
    def __init__(self):
        self.set_model_param()

    def train_interpolator(self, coord, value):
        logging.info(f"Building RBF with gaussan kernel on data shape {value.shape}")
        rbf_interpolator = RBFInterpolator(coord, value, kernel=self.kernel, epsilon=self.epsilon)
        return rbf_interpolator

    def set_model_param(self):
        self.kernel = "gaussian"
        self.epsilon = 0.5


    def build_rbf(self,  rbf_interpolator,  interp_scaler=None):
        if interp_scaler is None: interp_scaler = self.interp_scaler
        def rbf(x):
            flag = False
            if x.ndim == 1: 
                x = [x]
                flag = True
            x_scale = self.coord_scaler(x)
            interp = rbf_interpolator(x_scale)
            out = interp_scaler(interp)
            if flag: 
                return out[0]
            else:
                return out
        return rbf

    def build_logflux_rbf_interp(self, logflux):
        rbf_interp = self.train_interpolator(self.coord, logflux)
        rbf = self.build_rbf(rbf_interp)
        def interp_flux_fn(x, log=0):
            logflux = rbf(x)
            return logflux if log else np.exp(logflux)
        return interp_flux_fn


