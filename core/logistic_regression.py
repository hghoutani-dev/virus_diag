from sklearn.linear_model import LogisticRegression
from .model import Model
from utils.preprocessing import prepare_scaler

class LogisticRegressionModel(Model):
    def __init__(self, **kwargs):
        self.model = LogisticRegression(max_iter=1000, random_state=42, **kwargs)
        self.scaler = prepare_scaler()

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))
