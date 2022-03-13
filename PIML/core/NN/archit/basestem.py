from tensorflow import keras


class BaseStem():
    def __init__(self, input_dim, stem_dim) -> None:
        self.input_dim = input_dim
        self.stem_dim = stem_dim

    def add(self, layers):
        nn = keras.Sequential()
        for i, layer in enumerate(layers):
            nn.add(layer)
        return nn

    # def lay(self, model):
        

