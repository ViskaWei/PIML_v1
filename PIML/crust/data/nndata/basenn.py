import numpy as np
from abc import ABC, abstractmethod

class BaseNN(ABC):
    required_attributes = ["x_train", "y_train", "x_test", "y_test"]
    @abstractmethod
    def get_dim(self):
        pass

class NN(BaseNN):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.model = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test  = x_test
        self.y_test  = y_test
        self.get_dim()

        self.score   = None

    def get_dim(self):
        self.ntrain, self.dim = self.x_train.shape
        self.dim_out = self.y_train.shape[1]
        self.ntest  = self.x_test.shape[0]


class BaseNzNN(ABC):
    required_attributes = ["s_train", "s_test"]
    @abstractmethod
    def get_noise(self):
        pass

class NzNN(NN):
    def __init__(self, x_train, s_train, y_train, \
                x_test, s_test, y_test):
        super().__init__(x_train, y_train, x_test, y_test)
        self.s_train = s_train
        self.s_test  = s_test

    def get_noise(self, sigma):
        return np.random.normal(0, sigma, sigma.shape)
