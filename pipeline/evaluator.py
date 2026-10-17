import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import joblib

from core.dataset import PatientDataset
from utils.metrics import compute_metrics, display_metrics


class Evaluator:
    """
    Évaluateur indépendant de modèle IA sérialisé.
    
    Charge un modèle sauvegardé, l'applique sur les données de test
    et produit un bilan métrique complet.
    """

    def __init__(
        self,
        model_path: str | Path,
        data_path: Optional[str | Path] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.model_path = Path(model_path)
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        if not self.model_path.exists():
            raise FileNotFoundError(f"Fichier modèle introuvable : {self.model_path}")

    def run(self) -> Dict[str, Any]:
        """Charge le modèle et calcule les métriques sur le jeu de test."""
        print(f"\n[Evaluator] Chargement du modèle depuis : {self.model_path}")
        model = joblib.load(self.model_path)

        dataset = PatientDataset(self.data_path)
        _, X_test, _, y_test = dataset.split(
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=True
        )

        print(f"[Evaluator] Évaluation sur {len(X_test)} patients du jeu de test.")
        y_pred = model.predict(X_test)
        
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

        metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        display_metrics(metrics)
        return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évalue les performances d'un modèle sauvegardé.")
    parser.add_argument(
        "--model",
        default="models/logistic.joblib",
        help="Chemin vers le fichier de modèle .joblib (ex: models/logistic.joblib)"
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Chemin optionnel vers le fichier de données patient_data.csv"
    )
    args = parser.parse_args()

    evaluator = Evaluator(model_path=args.model, data_path=args.data)
    evaluator.run()
