from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError("predict_proba non disponible")
