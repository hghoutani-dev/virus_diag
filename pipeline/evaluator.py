import argparse
import joblib
from core.dataset import PatientDataset
from utils.metrics import classification_report_dict

def evaluate(model_path, data_path=None):
    model = joblib.load(model_path)
    _, X_test, _, y_test = PatientDataset(data_path).split()
    metrics = classification_report_dict(y_test, model.predict(X_test))
    print("=== Évaluation ===")
    for k,v in metrics.items():
        print(f"{k:10s}: {v:.4f}")
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default=None)
    a = p.parse_args()
    evaluate(a.model, a.data)
