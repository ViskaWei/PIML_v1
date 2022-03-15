from abc import ABC, abstractmethod
import logging
from tensorflow import keras

from PIML.core.NN.archit.basestem import BaseStem, DenseStem, ConvStem
from PIML.core.NN.archit.basebody import BaseBody, DenseBody, ConvBody


class BaseArchit(ABC):
    @abstractmethod
    def design(self):
        pass


class Archit(BaseArchit):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inputs = None
        self.stem = []
        self.body = []
        self.head = []

    def design(self):
        # inputs = keras.Input(shape = (self.input_dim, ), name='input')
        all_layers = [self.inputs, *self.stem, *self.body, *self.head]

        archit = keras.Sequential(all_layers)
        logging.info(archit.summary())
        return archit

    def design_stem(self, PARAM):
        stem_type = PARAM["type"]
        if stem_type == "dense":
            self.stem = DenseStem(self.input_dim, PARAM["act"], PARAM["dim"]).build()
        elif stem_type == "conv":
            self.stem = ConvStem(self.input_dim,  PARAM["act"], PARAM["filter"]).build()
        else:
            raise Exception("Unsupported Stem Type")
    

    def design_body(self, PARAM):
        body_type = PARAM["type"]
        if body_type == "dense":
            self.body = DenseBody(self.stem_dim, self.head_dim, PARAM["block"], PARAM["scheme"]).build()
        elif body_type == "conv":
            pass
            # self.body = ConvBody(PARAM["act"], PARAM["filter"]).build()
        else:
            raise Exception("Unsupported Body Type")

    def design_head(self, PARAM):
        pass

    # def get_units(self):
    #     if self.hidden_dims.size == 0:
    #         if self.input_dim <= 1000:
    #             hidden_dims = np.array([128, 64, 32, 16])
    #             # hidden_dims = np.array([128, 64, 32])
    #         elif self.input_dim < 2048:
    #             hidden_dims = np.array([1024, 512, 128, 32])
    #         else:
    #             hidden_dims = np.array([self.log2(2), self.log2(4), self.log2(8)])
    #         self.hidden_dims = hidden_dims
    #     self.hidden_dims = self.hidden_dims[self.hidden_dims > self.output_dim]
    #     units = [self.input_dim, *self.hidden_dims, self.output_dim]
    #     print(f"Layers: {units}")
    #     return units 

    # def add_dense_layer(self, regularizer=None):
    #     if regularizer is not None:
    #         kl1 = keras.regularizers.l1(regularizer)
    #     else:
    #         kl1 = None

    #     keras.layers.Dense(self.units[i + 1], kernel_regularizer=kl1, name=name)
                    


    # def add_dense_layer(self, unit, dp_rate=0., reg1=None, name=None):
    #     if reg1 is not None:
    #         kl1 = tf.keras.regularizers.l1(reg1)
    #     else:
    #         kl1 = None

    #     layer = keras.Sequential([keras.layers.Dense(unit, kernel_regularizer=kl1, name=name),
    #                 keras.layers.Dropout(dp_rate),
    #                 keras.layers.LeakyReLU(),
    #                 keras.layers.BatchNormalization(),
    #                 # keras.activations.tanh()
    #                 ])
    #     return layer