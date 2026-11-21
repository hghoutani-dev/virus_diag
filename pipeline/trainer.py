import argparse
from pathlib import Path
import joblib
from core.dataset import PatientDataset
from core.optimizer import build_model

def train(model_name="logistic", data_path=None, output_path="models/model.joblib"):
    ds = PatientDataset(data_path)
    X_train, X_test, y_train, y_test = ds.split()
    model = build_model(model_name)
    model.fit(X_train, y_train)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"Modèle sauvegardé : {out}")
    print(f"Train={len(X_train)}, Test={len(X_test)}")
    return model

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["logistic","neural_network"], default="logistic")
    p.add_argument("--data", default=None)
    p.add_argument("--output", default="models/model.joblib")
    a = p.parse_args()
    train(a.model, a.data, a.output)
