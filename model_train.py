"""
Script d'exemple d'entraînement d'un modèle pour virus_diag (Ticket #1)
Permet d'entraîner et de sérialiser un modèle de diagnostic.
"""

from pathlib import Path
import joblib
from core.dataset import PatientDataset
from core.optimizer import build_model
from utils.metrics import classification_report_dict


def train_example_model(model_name: str = "logistic", output_path: str = "models/model_train.joblib"):
    print("=" * 60)
    print(f" [ENTRAÎNEMENT] Modèle d'exemple : {model_name.upper()}")
    print("=" * 60)

    # 1. Chargement et séparation des données
    data_file = Path("data") / "patient_data.csv"
    dataset = PatientDataset(str(data_file) if data_file.exists() else None)
    X_train, X_test, y_train, y_test = dataset.split()

    print(f">> Données chargées : {len(X_train)} exemples train, {len(X_test)} exemples test")

    # 2. Construction et ajustement du modèle
    model = build_model(model_name)
    print(">> Entraînement du modèle en cours...")
    model.fit(X_train, y_train)

    # 3. Évaluation sur l'ensemble de test
    y_pred = model.predict(X_test)
    metrics = classification_report_dict(y_test, y_pred)
    print("\n>> Métriques obtenues sur le jeu de test :")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   - {k}: {v:.4f}")
        else:
            print(f"   - {k}: {v}")

    # 4. Sauvegarde du modèle entraîné
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"\n>> Modèle d'exemple sauvegardé avec succès dans : {out}\n")
    return model


if __name__ == "__main__":
    train_example_model()
