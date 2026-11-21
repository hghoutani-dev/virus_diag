import argparse
from pathlib import Path
import joblib
import pandas as pd

class ClinicalPredictor:
    """Adaptateur clinique : indépendant du type concret de modèle IA.

    Le modèle injecté doit fournir predict(). Si predict_proba() existe,
    la probabilité de la classe 1 est comparée au seuil 0.5.
    """

    def __init__(self, model, threshold=.5):
        if not hasattr(model, "predict"):
            raise TypeError("Le modèle doit posséder predict().")
        self.model = model
        self.threshold = threshold

    @staticmethod
    def _dataframe(patient_data):
        if isinstance(patient_data, pd.DataFrame):
            return patient_data
        if isinstance(patient_data, dict):
            return pd.DataFrame([patient_data])
        return pd.DataFrame(patient_data)

    def _score(self, patient_data):
        X = self._dataframe(patient_data)
        if hasattr(self.model, "predict_proba"):
            try:
                return float(self.model.predict_proba(X)[0][1])
            except (NotImplementedError, AttributeError, IndexError):
                pass
        pred = self.model.predict(X)
        return float(pred[0] if hasattr(pred, "__len__") else pred)

    def diagnose(self, patient_data):
        return "Infecté" if self._score(patient_data) >= self.threshold else "Sain"

    def diagnose_with_score(self, patient_data):
        score = self._score(patient_data)
        return {"diagnosis": "Infecté" if score >= self.threshold else "Sain",
                "probability": score}

def demo(model_path):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"{model_path} absent : entraînez le modèle d'abord.")
    predictor = ClinicalPredictor(joblib.load(model_path))
    patient = {
        "age":45, "temperature":39.0, "cough":1, "fatigue":1,
        "sore_throat":1, "breathing_difficulty":0,
        "oxygen_saturation":96, "heart_rate":98
    }
    print("=== Diagnostic clinique ===")
    print(predictor.diagnose_with_score(patient))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/logistic.joblib")
    a = p.parse_args()
    demo(a.model)
