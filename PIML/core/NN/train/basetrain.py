

from abc import ABC, abstractmethod

class BaseTrain(ABC):
    @abstractmethod
    def complie(self):
        pass
    @abstractmethod
    def train(self):
        pass


class Train(BaseTrain):
    def __init__(self, batch_size, epochs, validation_split) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

    def train(self, model, x_train, y_train):
        model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)