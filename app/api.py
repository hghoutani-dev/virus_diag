from flask import Flask, jsonify, request
import joblib
from .interface_clinique import ClinicalPredictor

app = Flask(__name__)
predictor = None

@app.get("/health")
def health():
    return jsonify({"status":"ok"})

@app.post("/predict")
def predict():
    global predictor
    if predictor is None:
        predictor = ClinicalPredictor(joblib.load("models/logistic.joblib"))
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error":"JSON patient attendu"}), 400
    try:
        return jsonify(predictor.diagnose_with_score(data))
    except Exception as e:
        return jsonify({"error":str(e)}), 400

if __name__ == "__main__":
    predictor = ClinicalPredictor(joblib.load("models/logistic.joblib"))
    app.run(host="0.0.0.0", port=5000)
