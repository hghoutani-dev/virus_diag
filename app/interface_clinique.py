import argparse
from pathlib import Path
from typing import Any, Dict, Union
import joblib
import numpy as np
import pandas as pd

from utils.preprocessing import ALL_FEATURES


class ClinicalPredictor:
    """
    Interface applicative et clinique pour le diagnostic viral.
    
    Principes de conception POO :
    - Responsabilité Unique (SRP) : ClinicalPredictor n'entraîne JAMAIS de modèle.
      Il se focalise exclusivement sur l'inférence clinique à partir de données patient.
    - Polymorphisme et Duck Typing : Reçoit n'importe quel modèle IA déjà entraîné
      (LogisticRegressionModel, NeuralNetworkModel, ou tout objet exposant 'predict()').
    - Règle de décision médicale :
      * Si probabilité >= seuil (0.5 par défaut) -> "Infecté"
      * Sinon -> "Sain"
    """

    def __init__(self, model: Any, threshold: float = 0.5):
        """
        Initialise le prédicteur clinique avec un modèle pré-entraîné injecté.
        
        Args:
            model: Objet modèle IA disposant au minimum d'une méthode predict().
            threshold: Seuil de décision clinique (défaut: 0.5).
            
        Raises:
            TypeError: Si le modèle ne respecte pas l'interface minimale requise.
        """
        if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
            raise TypeError("Le modèle injecté doit obligatoirement posséder une méthode callable 'predict()'.")

        self.model = model
        self.threshold = float(threshold)

    def _prepare_input(self, patient_data: Union[Dict[str, Any], pd.DataFrame, list]) -> pd.DataFrame:
        """
        Convertit et valide les données cliniques d'un patient sous forme de DataFrame pandas.
        """
        if isinstance(patient_data, pd.DataFrame):
            df = patient_data.copy()
        elif isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        elif isinstance(patient_data, list):
            df = pd.DataFrame(patient_data)
        else:
            raise TypeError("Format de données patient non supporté. Attendu: dict, list ou pd.DataFrame.")

        # Vérification des colonnes nécessaires
        missing = [col for col in ALL_FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"Données patient incomplètes. Variables manquantes : {missing}")

        return df[ALL_FEATURES]

    def get_probability(self, patient_data: Union[Dict[str, Any], pd.DataFrame]) -> float:
        """
        Estime la probabilité d'infection du patient.
        
        Stratégie :
        1. Si le modèle dispose de predict_proba(), on extrait la probabilité de la classe 1 (Infecté).
        2. Si predict_proba() n'est pas disponible ou lève une exception, on se rabat sur predict() :
           le score binaire (0 ou 1) est alors converti en probabilité discrète (0.0 ou 1.0).
        """
        X = self._prepare_input(patient_data)

        # Tentative avec predict_proba
        if hasattr(self.model, "predict_proba") and callable(getattr(self.model, "predict_proba")):
            try:
                proba = self.model.predict_proba(X)
                # Format scikit-learn standard: matrice (n_samples, 2)
                if hasattr(proba, "ndim") and proba.ndim == 2 and proba.shape[1] >= 2:
                    return float(proba[0, 1])
                elif hasattr(proba, "__len__") and len(proba) > 0:
                    first = proba[0]
                    if hasattr(first, "__len__") and len(first) >= 2:
                        return float(first[1])
                    return float(first)
            except (NotImplementedError, AttributeError):
                pass  # Fallback sur predict()

        # Fallback si predict_proba() n'est pas implémenté
        pred = self.model.predict(X)
        val = pred[0] if hasattr(pred, "__len__") else pred
        return 1.0 if float(val) >= 0.5 else 0.0

    def diagnose(self, patient_data: Union[Dict[str, Any], pd.DataFrame]) -> str:
        """
        Émet un diagnostic textuel selon le seuil décisionnel.
        
        Returns:
            "Infecté" si probabilité >= seuil, sinon "Sain".
        """
        probability = self.get_probability(patient_data)
        return "Infecté" if probability >= self.threshold else "Sain"

    def diagnose_with_score(self, patient_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Fournit un compte-rendu médical complet comprenant le diagnostic et la probabilité.
        
        Returns:
            dict: {"diagnosis": "Infecté" | "Sain", "probability": float}
        """
        probability = self.get_probability(patient_data)
        diagnosis = "Infecté" if probability >= self.threshold else "Sain"
        return {
            "diagnosis": diagnosis,
            "probability": round(probability, 4)
        }


def demo(model_path: str = "models/logistic.joblib") -> None:
    """Exemple d'exécution clinique en ligne de commande."""
    resolved_path = Path(model_path)
    if not resolved_path.is_absolute():
        resolved_path = Path(__file__).resolve().parents[1] / model_path

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable ({resolved_path}). Veuillez d'abord l'entraîner via pipeline.trainer."
        )

    model = joblib.load(resolved_path)
    predictor = ClinicalPredictor(model=model, threshold=0.5)

    sample_patient = {
        "age": 67,
        "temperature": 38.9,
        "cough": 1,
        "fatigue": 1,
        "sore_throat": 1,
        "breathing_difficulty": 1,
        "oxygen_saturation": 91.5,
        "heart_rate": 105
    }

    result = predictor.diagnose_with_score(sample_patient)
    print("=" * 45)
    print("      DÉMONSTRATION DU DIAGNOSTIC CLINIQUE")
    print("=" * 45)
    print("Paramètres du patient testé :")
    for k, v in sample_patient.items():
        print(f"  - {k:22s} : {v}")
    print("-" * 45)
    print(f"Diagnostic émis  : {result['diagnosis']}")
    print(f"Probabilité      : {result['probability'] * 100:.2f}%")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test clinique d'un patient.")
    parser.add_argument("--model", default="models/logistic.joblib", help="Chemin du modèle .joblib")
    args = parser.parse_args()
    demo(args.model)
