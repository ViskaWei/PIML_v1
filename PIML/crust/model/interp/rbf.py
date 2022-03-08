
import numpy as np
from scipy.interpolate import RBFInterpolator

class RBF(object):
    def __init__(self, coord, coord_scaler=None, interp_scaler=None):
        self.rbf = None
        self.coord = coord
        self.coord_scaler = coord_scaler

        if coord_scaler is None:
            self.coord_scaler = lambda x: x
        
        if interp_scaler is None:
            self.interp_scaler = lambda x: x

    def train_rbf(self, coord, val):
        print(f"Building RBF with gaussan kernel on data shape {val.shape}")
        rbf_interpolator = RBFInterpolator(coord, val, kernel='gaussian', epsilon=0.5)
        return rbf_interpolator

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
        rbf_interp = self.train_rbf(self.coord, logflux)
        rbf = self.build_rbf(rbf_interp)
        def interp_flux_fn(x, log=0):
            logflux = rbf(x)
            return logflux if log else np.exp(logflux)
        return interp_flux_fn


