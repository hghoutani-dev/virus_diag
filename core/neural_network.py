from sklearn.neural_network import MLPClassifier
from .model import Model
from utils.preprocessing import prepare_scaler

class NeuralNetworkModel(Model):
    def __init__(self, **kwargs):
        defaults = dict(hidden_layer_sizes=(32,16), activation="relu",
                        solver="adam", max_iter=1000,
                        random_state=42, early_stopping=True)
        defaults.update(kwargs)
        self.model = MLPClassifier(**defaults)
        self.scaler = prepare_scaler()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))
