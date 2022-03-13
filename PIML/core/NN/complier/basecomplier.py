
from abc import ABC, abstractmethod


class BaseComplier(ABC):
    @abstractmethod
    def complie(self):
        pass


class Complier(BaseComplier):
    def __init__(self, loss, optimzer, metrics):
        self.loss = loss
        self.optimzer = optimzer
        self.metrics = metrics

    def complie(self, model):
        model.compile(loss=self.loss, optimizer=self.optimzer, metrics=self.metrics)