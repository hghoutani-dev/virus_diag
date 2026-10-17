from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd

# Colonnes identifiées dans patient_data.csv
NUMERICAL_FEATURES = [
    "age",
    "temperature",
    "oxygen_saturation",
    "heart_rate"
]

BINARY_FEATURES = [
    "cough",
    "fatigue",
    "sore_throat",
    "breathing_difficulty"
]

ALL_FEATURES = [
    "age",
    "temperature",
    "cough",
    "fatigue",
    "sore_throat",
    "breathing_difficulty",
    "oxygen_saturation",
    "heart_rate"
]

TARGET_COLUMN = "infected"


def create_preprocessor() -> ColumnTransformer:
    """
    Construit un ColumnTransformer adapté à la structure de patient_data.csv.
    
    - Variables numériques continues (age, temperature, oxygen_saturation, heart_rate) :
      Imputation par la médiane + Standardisation (StandardScaler).
    - Variables binaires / symptômes (cough, fatigue, sore_throat, breathing_difficulty) :
      Imputation par la valeur la plus fréquente (0 par défaut) + conservation brute (passthrough).
      
    Returns:
        ColumnTransformer configuré pour traiter X.
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    bin_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("bin", bin_pipeline, BINARY_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor
