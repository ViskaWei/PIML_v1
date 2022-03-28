from PIML.crust.data.nndata.basenn import NN
from PIML.gateway.loaderIF.baseloaderIF import BaseLoaderIF
from PIML.surface.database.nnloader import MINSTDataLoader



class NNDataLoaderIF(BaseLoaderIF):
    """ class for loading NN Data from keras.dataset . """
    def set_loader(self, name):
        if name =="MINST":
            loader = MINSTDataLoader()
        else:
            raise ValueError("Unknown NN name")
        return loader

    def load(self, name: str):
        '''
        Output: x_train, y_train, x_test, y_test
        '''
        loader = self.set_loader(name)
        x_train, y_train, x_test, y_test = loader.load()
        return NN(x_train, y_train, x_test, y_test)

