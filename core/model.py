from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd


class Model(ABC):
    """
    Classe de base abstraite représentant un modèle de diagnostic d'IA.
    
    Définit le contrat d'interface commun que tous les modèles concrets
    (Régression Logistique, Réseau de Neurones, etc.) doivent respecter.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "Model":
        """
        Entraîne le modèle à partir des variables explicatives X et de la cible y.
        
        Args:
            X: Données d'entraînement (features).
            y: Variable cible (étiquettes 0 ou 1).
            
        Returns:
            self: L'instance du modèle entraînée.
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Prédit la classe binaire (0 = Sain, 1 = Infecté) pour chaque échantillon.
        
        Args:
            X: Données de test ou données cliniques du patient.
            
        Returns:
            Tableau numpy contenant les prédictions (0 ou 1).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Prédit les probabilités d'appartenance aux classes [P(Sain), P(Infecté)].
        
        Args:
            X: Données de test ou données cliniques du patient.
            
        Returns:
            Tableau numpy de forme (n_samples, 2) avec les probabilités.
        """
        pass
