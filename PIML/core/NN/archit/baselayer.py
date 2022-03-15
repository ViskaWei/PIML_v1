from keras import layers
from keras import activations

class BaseLayer():

    def lay_dense(self, dim):
        return layers.Dense(dim)

    def lay_batchnorm(self):
        return layers.BatchNormalization()

    def lay_dropout(self, rate):
        return layers.Dropout(rate)

    def lay_conv(self, *args):
        return layers.Conv2D(*args)

    def lay_act(self, name, *args):
        if name == "relu":
            return layers.Activation(activations.relu)
        elif name == "tanh":
            return layers.Activation(activations.tanh)
        elif name == "sigmoid":
            return layers.Activation(activations.sigmoid)
        elif name == "softmax":
            '''
            Softmax converts a vector of values to a probability distribution.
            axis = -1
            '''
            return layers.Activation(activations.softmax, *args)
        elif name == "linear":
            return layers.Activation(activations.linear)
        elif name == "leaky":
            # alpha = 0.3
            return layers.LeakyReLU(*args)
