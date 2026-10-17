from pathlib import Path
from typing import Optional, Dict
import joblib
from flask import Flask, jsonify, request, render_template

from app.interface_clinique import ClinicalPredictor
from utils.preprocessing import ALL_FEATURES

app = Flask(__name__, template_folder="templates")

# Cache pour les prédicteurs selon l'architecture (logistic ou neural_network)
_predictors_cache: Dict[str, ClinicalPredictor] = {}
_default_model_name = "logistic"


def get_predictor(model_name: Optional[str] = None) -> ClinicalPredictor:
    """Charge ou retourne l'instance de ClinicalPredictor en cache pour le modèle demandé."""
    global _predictors_cache
    target_name = (model_name or _default_model_name).lower()
    
    if "neural" in target_name or "mlp" in target_name:
        key = "neural_network"
    else:
        key = "logistic"

    if key not in _predictors_cache:
        models_dir = Path(__file__).resolve().parents[1] / "models"
        model_file = models_dir / f"{key}.joblib"

        if not model_file.exists():
            # Fallback sur n'importe quel autre modèle existant
            alt = "neural_network" if key == "logistic" else "logistic"
            model_file = models_dir / f"{alt}.joblib"

        if not model_file.exists():
            raise FileNotFoundError(
                "Aucun modèle entraîné trouvé dans 'models/'. "
                "Exécutez 'python -m pipeline.trainer --model logistic' pour créer le modèle."
            )

        model = joblib.load(model_file)
        _predictors_cache[key] = ClinicalPredictor(model=model, threshold=0.5)

    return _predictors_cache[key]


@app.get("/")
def index():
    """Point d'entrée web : affiche l'interface clinique interactive."""
    return render_template("index.html")


@app.get("/health")
def health():
    """Vérification de l'état de l'API et des modèles IA disponibles."""
    try:
        predictor = get_predictor()
        return jsonify({
            "status": "ok",
            "active_model": type(predictor.model).__name__,
            "expected_features": ALL_FEATURES
        }), 200
    except Exception as e:
        return jsonify({
            "status": "degraded",
            "error": str(e),
            "expected_features": ALL_FEATURES
        }), 200


@app.post("/predict")
def predict():
    """
    Point de terminaison clinique pour diagnostiquer un patient.
    
    Prend en charge le choix du modèle via paramètre d'URL (ex: ?model=neural_network).
    """
    if not request.is_json:
        return jsonify({
            "error": "Requête invalide : le contenu doit être au format application/json."
        }), 400

    payload = request.get_json()

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Le corps de la requête doit être un objet JSON contenant les variables du patient."
        }), 400

    # Validation de la présence de toutes les caractéristiques requises
    missing = [f for f in ALL_FEATURES if f not in payload]
    if missing:
        return jsonify({
            "error": f"Caractéristiques manquantes dans les données du patient : {missing}",
            "required_features": ALL_FEATURES
        }), 400

    # Validation du typage numérique
    for feature in ALL_FEATURES:
        val = payload[feature]
        if not isinstance(val, (int, float)):
            return jsonify({
                "error": f"La caractéristique '{feature}' doit être une valeur numérique (reçu: {type(val).__name__})."
            }), 400

    model_arg = request.args.get("model", _default_model_name)

    try:
        predictor = get_predictor(model_arg)
        result = predictor.diagnose_with_score(payload)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            "error": f"Erreur lors de l'inférence : {str(e)}"
        }), 500


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("      DÉMARRAGE DE L'INTERFACE CLINIQUE VIRUSDIAG")
    print("=" * 60)
    try:
        get_predictor("logistic")
        get_predictor("neural_network")
        print("[API] Modèles 'logistic' et 'neural_network' chargés avec succès.")
    except Exception as exc:
        print(f"[API] Note de démarrage : {exc}")

    print("\n  >> Interface web accessible sur : http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
