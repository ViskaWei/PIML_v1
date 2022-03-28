import os
from PIML.crust.data.nndata.basenznn import NzNN
from PIML.gateway.loaderIF.baseloaderIF import ParamLoaderIF, DictLoaderIF


class StellarNNLoaderIF(ParamLoaderIF):
    def set_param(self, PARAM):
        self.dir         = PARAM["path"]
        self.train_path  = os.path.join(self.dir, PARAM["train_name"])
        self.test_path   = os.path.join(self.dir, PARAM["test_name"])

    def load(self):
        train = self.load_data(self.train_path)
        test  = self.load_data(self.test_path)
        return NzNN(train["data"], train["sigma"], train["label"],\
                    test["data"] , test["sigma"] , test["label"])

    def load_data(self, path):
        loader = DictLoaderIF()
        loader.set_path(path)
        return loader.load_dict_args()

