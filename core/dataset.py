from pathlib import Path
import pandas as pd

FEATURES = ["age","temperature","cough","fatigue","sore_throat",
            "breathing_difficulty","oxygen_saturation","heart_rate"]
TARGET = "infected"

class PatientDataset:
    def __init__(self, path=None):
        default = Path(__file__).resolve().parents[1] / "data" / "patient_data.csv"
        self.path = Path(path) if path else default
        self.data = pd.read_csv(self.path)

    def features(self):
        return self.data[FEATURES].copy()

    def target(self):
        return self.data[TARGET].copy()

    def split(self, test_size=.2, random_state=42):
        from sklearn.model_selection import train_test_split
        return train_test_split(
            self.features(), self.target(), test_size=test_size,
            random_state=random_state, stratify=self.target()
        )
