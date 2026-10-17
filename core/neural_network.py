from typing import Any
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .model import Model
from utils.preprocessing import create_preprocessor


class NeuralNetworkModel(Model):
    """
    Modèle de classification par Réseau de Neurones Artificiels (Perceptron Multicouche - MLP).
    
    Encapsule la chaîne complète de prétraitement (ColumnTransformer)
    et l'estimateur MLPClassifier au sein d'un Pipeline scikit-learn.
    Le prétraitement (StandardScaler sur les variables continues)
    est critique pour la convergence de la descente de gradient dans un MLP.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialise le réseau de neurones avec son préprocesseur.
        
        Args:
            **kwargs: Hyperparamètres passés à l'estimateur MLPClassifier.
        """
        defaults = {
            "hidden_layer_sizes": (32, 16),
            "activation": "relu",
            "solver": "adam",
            "learning_rate_init": 0.005,
            "max_iter": 1000,
            "random_state": 42
        }
        defaults.update(kwargs)

        self.preprocessor = create_preprocessor()
        self.classifier = MLPClassifier(**defaults)
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", self.classifier)
        ])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "NeuralNetworkModel":
        """
        Ajuste le préprocesseur et entraîne le réseau de neurones.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Prédit la classe binaire (0 = Sain, 1 = Infecté) pour chaque échantillon.
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités d'appartenance aux classes [P(Sain), P(Infecté)].
        """
        return self.pipeline.predict_proba(X)
