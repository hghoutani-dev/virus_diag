import unittest
import numpy as np
from app.interface_clinique import ClinicalPredictor
from core.dataset import PatientDataset
from core.optimizer import build_model


class DummyPredictOnlyModel:
    """Modèle factice sans predict_proba pour tester le polymorphisme et le fallback."""
    def __init__(self, return_class=1):
        self.return_class = return_class

    def predict(self, X):
        return np.array([self.return_class] * len(X))


class TestClinicalPredictor(unittest.TestCase):
    """Vérifie le fonctionnement de ClinicalPredictor, le seuil 0.5 et le polymorphisme."""

    @classmethod
    def setUpClass(cls):
        dataset = PatientDataset()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = dataset.split(test_size=0.2, random_state=42)

        # Entraînement des deux modèles
        cls.log_model = build_model("logistic").fit(cls.X_train, cls.y_train)
        cls.nn_model = build_model("neural_network").fit(cls.X_train, cls.y_train)

        cls.sample_infected = {
            "age": 75,
            "temperature": 39.5,
            "cough": 1,
            "fatigue": 1,
            "sore_throat": 1,
            "breathing_difficulty": 1,
            "oxygen_saturation": 91.0,
            "heart_rate": 110
        }

        cls.sample_healthy = {
            "age": 25,
            "temperature": 36.6,
            "cough": 0,
            "fatigue": 0,
            "sore_throat": 0,
            "breathing_difficulty": 0,
            "oxygen_saturation": 98.5,
            "heart_rate": 70
        }

    def test_invalid_model_type(self):
        class InvalidObject:
            pass

        with self.assertRaises(TypeError):
            ClinicalPredictor(InvalidObject())

    def test_polymorphism_with_both_models(self):
        predictor_log = ClinicalPredictor(self.log_model)
        predictor_nn = ClinicalPredictor(self.nn_model)

        diag_log = predictor_log.diagnose(self.sample_infected)
        diag_nn = predictor_nn.diagnose(self.sample_infected)

        self.assertIn(diag_log, ["Infecté", "Sain"])
        self.assertIn(diag_nn, ["Infecté", "Sain"])

    def test_threshold_behavior(self):
        # Modèle avec predict_proba mocké via classe ou fonction
        class ControlledProbabilityModel:
            def __init__(self, proba_infected):
                self.p = proba_infected

            def predict(self, X):
                return np.array([1 if self.p >= 0.5 else 0])

            def predict_proba(self, X):
                return np.array([[1.0 - self.p, self.p]])

        # Test frontière seuil 0.5
        pred_high = ClinicalPredictor(ControlledProbabilityModel(0.50), threshold=0.5)
        self.assertEqual(pred_high.diagnose(self.sample_healthy), "Infecté")

        pred_above = ClinicalPredictor(ControlledProbabilityModel(0.75), threshold=0.5)
        self.assertEqual(pred_above.diagnose(self.sample_healthy), "Infecté")

        pred_below = ClinicalPredictor(ControlledProbabilityModel(0.49), threshold=0.5)
        self.assertEqual(pred_below.diagnose(self.sample_healthy), "Sain")

    def test_fallback_when_no_predict_proba(self):
        dummy_infected = DummyPredictOnlyModel(return_class=1)
        predictor = ClinicalPredictor(dummy_infected)
        result = predictor.diagnose_with_score(self.sample_healthy)
        self.assertEqual(result["diagnosis"], "Infecté")
        self.assertEqual(result["probability"], 1.0)

        dummy_healthy = DummyPredictOnlyModel(return_class=0)
        predictor_healthy = ClinicalPredictor(dummy_healthy)
        result_h = predictor_healthy.diagnose_with_score(self.sample_healthy)
        self.assertEqual(result_h["diagnosis"], "Sain")
        self.assertEqual(result_h["probability"], 0.0)

    def test_diagnose_with_score_structure(self):
        predictor = ClinicalPredictor(self.log_model)
        result = predictor.diagnose_with_score(self.sample_healthy)
        self.assertIn("diagnosis", result)
        self.assertIn("probability", result)
        self.assertIsInstance(result["diagnosis"], str)
        self.assertIsInstance(result["probability"], float)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
