import unittest
from core.dataset import PatientDataset
from utils.preprocessing import ALL_FEATURES, TARGET_COLUMN


class TestPatientDataset(unittest.TestCase):
    """Vérifie le chargement, la validation et le découpage des données."""

    def setUp(self):
        self.dataset = PatientDataset()

    def test_dataset_loading(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertEqual(len(self.dataset.data), 600)
        self.assertEqual(len(self.dataset.data.columns), 9)

    def test_features_and_target_columns(self):
        features = self.dataset.features()
        target = self.dataset.target()
        
        self.assertEqual(list(features.columns), ALL_FEATURES)
        self.assertEqual(target.name, TARGET_COLUMN)
        self.assertEqual(len(features), len(target))

    def test_dataset_split_stratified(self):
        X_train, X_test, y_train, y_test = self.dataset.split(test_size=0.2, random_state=42)
        
        self.assertEqual(len(X_train), 480)
        self.assertEqual(len(X_test), 120)
        
        # Vérification du maintien des proportions de la cible (stratification)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        self.assertAlmostEqual(train_ratio, test_ratio, delta=0.03)


if __name__ == "__main__":
    unittest.main()
