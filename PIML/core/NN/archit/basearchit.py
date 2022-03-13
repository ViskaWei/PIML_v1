from abc import ABC, abstractmethod
import logging
from tensorflow import keras
# import tensorflow as tf


class BaseArchit(ABC):
    @abstractmethod
    def design(self):
        pass


class Archit(BaseArchit):
    def __init__(self):

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



class DNNArchit(BaseArchit):

    def __init__(self, input_dim, output_dim) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        
    

            
            


    def get_units(self):
        if self.hidden_dims.size == 0:
            if self.input_dim <= 1000:
                hidden_dims = np.array([128, 64, 32, 16])
                # hidden_dims = np.array([128, 64, 32])
            elif self.input_dim < 2048:
                hidden_dims = np.array([1024, 512, 128, 32])
            else:
                hidden_dims = np.array([self.log2(2), self.log2(4), self.log2(8)])
            self.hidden_dims = hidden_dims
        self.hidden_dims = self.hidden_dims[self.hidden_dims > self.output_dim]
        units = [self.input_dim, *self.hidden_dims, self.output_dim]
        print(f"Layers: {units}")
        return units 

    def add_dense_layer(self, regularizer=None):
        if regularizer is not None:
            kl1 = keras.regularizers.l1(regularizer)
        else:
            kl1 = None

        keras.layers.Dense(self.units[i + 1], kernel_regularizer=kl1, name=name)
                    


    def add_dense_layer(self, unit, dp_rate=0., reg1=None, name=None):
        if reg1 is not None:
            kl1 = tf.keras.regularizers.l1(reg1)
        else:
            kl1 = None

        layer = keras.Sequential([keras.layers.Dense(unit, kernel_regularizer=kl1, name=name),
                    keras.layers.Dropout(dp_rate),
                    keras.layers.LeakyReLU(),
                    keras.layers.BatchNormalization(),
                    # keras.activations.tanh()
                    ])
        return layer