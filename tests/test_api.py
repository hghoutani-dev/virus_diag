import unittest
import json
from app.api import app
from pipeline.trainer import Trainer


class TestAPI(unittest.TestCase):
    """Vérifie le comportement des points de terminaison Flask."""

    @classmethod
    def setUpClass(cls):
        # Assure qu'au moins un modèle est entraîné et disponible dans models/
        trainer = Trainer(model_name="logistic")
        trainer.run()
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("expected_features", data)

    def test_predict_endpoint_success(self):
        patient_data = {
            "age": 55,
            "temperature": 38.6,
            "cough": 1,
            "fatigue": 1,
            "sore_throat": 0,
            "breathing_difficulty": 1,
            "oxygen_saturation": 93.5,
            "heart_rate": 88
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(patient_data),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("diagnosis", data)
        self.assertIn(data["diagnosis"], ["Infecté", "Sain"])
        self.assertIn("probability", data)
        self.assertIsInstance(data["probability"], float)

    def test_predict_missing_feature(self):
        # Manque oxygen_saturation
        incomplete_patient = {
            "age": 55,
            "temperature": 38.6,
            "cough": 1,
            "fatigue": 1,
            "sore_throat": 0,
            "breathing_difficulty": 1,
            "heart_rate": 88
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(incomplete_patient),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_predict_non_json(self):
        response = self.client.post(
            "/predict",
            data="plain text",
            content_type="text/plain"
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
