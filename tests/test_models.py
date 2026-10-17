import unittest
import numpy as np
from core.dataset import PatientDataset
from core.optimizer import build_model
from core.model import Model
from core.logistic_regression import LogisticRegressionModel
from core.neural_network import NeuralNetworkModel


class TestModels(unittest.TestCase):
    """Vérifie l'instanciation, l'entraînement et la prédiction des modèles IA."""

    @classmethod
    def setUpClass(cls):
        dataset = PatientDataset()
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = dataset.split(test_size=0.2, random_state=42)

    def test_factory_build_model(self):
        log_model = build_model("logistic")
        nn_model = build_model("neural_network")

        self.assertIsInstance(log_model, Model)
        self.assertIsInstance(log_model, LogisticRegressionModel)
        self.assertIsInstance(nn_model, Model)
        self.assertIsInstance(nn_model, NeuralNetworkModel)

        with self.assertRaises(ValueError):
            build_model("unsupported_model_type")

    def test_logistic_regression_fit_predict(self):
        model = build_model("logistic", max_iter=200)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        probas = model.predict_proba(self.X_test)

        self.assertEqual(len(preds), len(self.X_test))
        self.assertEqual(probas.shape, (len(self.X_test), 2))
        self.assertTrue(np.all((preds == 0) | (preds == 1)))
        self.assertTrue(np.all(probas >= 0.0) and np.all(probas <= 1.0))

    def test_neural_network_fit_predict(self):
        model = build_model("neural_network", max_iter=100)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_test)
        probas = model.predict_proba(self.X_test)

        self.assertEqual(len(preds), len(self.X_test))
        self.assertEqual(probas.shape, (len(self.X_test), 2))
        self.assertTrue(np.all((preds == 0) | (preds == 1)))
        self.assertTrue(np.all(probas >= 0.0) and np.all(probas <= 1.0))


if __name__ == "__main__":
    unittest.main()
