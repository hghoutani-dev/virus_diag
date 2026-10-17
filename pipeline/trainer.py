import argparse
from pathlib import Path
from typing import Optional
import joblib

from core.dataset import PatientDataset
from core.optimizer import build_model
from core.model import Model
from utils.metrics import compute_metrics, display_metrics


class Trainer:
    """
    Orchestrateur responsable du cycle complet d'entraînement et de sauvegarde.
    
    Respecte le principe de responsabilité unique (SRP) :
    le Trainer est uniquement dédié à la préparation des données,
    l'entraînement, l'évaluation de validation et la persistance du modèle.
    """

    def __init__(
        self,
        model_name: str = "logistic",
        data_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.model_name = model_name
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        # Détermination du chemin de sortie par défaut : models/{model_name}.joblib
        if output_path is None:
            models_dir = Path(__file__).resolve().parents[1] / "models"
            normalized_name = "logistic" if "logistic" in model_name else "neural_network"
            self.output_path = models_dir / f"{normalized_name}.joblib"
        else:
            self.output_path = Path(output_path)

    def run(self) -> Model:
        """
        Exécute la chaîne d'entraînement :
        1. Chargement des données
        2. Découpage stratifié train/test
        3. Instanciation du modèle (avec préprocesseur encapsulé)
        4. Entraînement (.fit)
        5. Évaluation sur l'échantillon de test
        6. Sauvegarde du modèle sérialisé (.joblib)
        """
        print(f"\n[Trainer] Démarrage de l'entraînement du modèle : '{self.model_name}'")
        dataset = PatientDataset(self.data_path)
        X_train, X_test, y_train, y_test = dataset.split(
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=True
        )

        print(f"[Trainer] Données chargées : Train={len(X_train)} patients, Test={len(X_test)} patients.")

        # Construction du modèle via la fabrique
        model = build_model(self.model_name)

        # Entraînement
        print("[Trainer] Ajustement du préprocesseur et entraînement du classifieur...")
        model.fit(X_train, y_train)

        # Évaluation sur le jeu de test
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

        metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)
        print("\n[Trainer] Métriques obtenues sur le jeu de test :")
        display_metrics(metrics)

        # Sauvegarde
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.output_path)
        print(f"[Trainer] Modèle sauvegardé avec succès dans : {self.output_path}\n")

        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne et sauvegarde un modèle de diagnostic.")
    parser.add_argument(
        "--model",
        choices=["logistic", "neural_network"],
        default="logistic",
        help="Type d'architecture de modèle à entraîner (défaut: logistic)"
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Chemin vers le fichier patient_data.csv (optionnel)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Chemin de sauvegarde du fichier .joblib (défaut: models/<model>.joblib)"
    )
    args = parser.parse_args()

    trainer = Trainer(
        model_name=args.model,
        data_path=args.data,
        output_path=args.output
    )
    trainer.run()
