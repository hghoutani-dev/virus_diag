from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .model import Model
from utils.preprocessing import create_preprocessor


class LogisticRegressionModel(Model):
    """
    Modèle de classification par Régression Logistique.
    
    Encapsule la chaîne complète de prétraitement (ColumnTransformer)
    et l'estimateur LogisticRegression au sein d'un Pipeline scikit-learn.
    Ainsi, le prétraitement est sauvegardé et sérialisé conjointement avec le modèle.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialise le modèle de régression logistique avec son préprocesseur.
        
        Args:
            **kwargs: Hyperparamètres passés à l'estimateur LogisticRegression.
        """
        defaults = {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced"  # Utile face au déséquilibre de classes (80% / 20%)
        }
        defaults.update(kwargs)

        self.preprocessor = create_preprocessor()
        self.classifier = LogisticRegression(**defaults)
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("classifier", self.classifier)
        ])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "LogisticRegressionModel":
        """
        Ajuste le préprocesseur et entraîne la régression logistique.
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
