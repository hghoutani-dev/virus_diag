from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocessing import ALL_FEATURES, TARGET_COLUMN


class PatientDataset:
    """
    Gestionnaire orienté objet du jeu de données des patients (patient_data.csv).
    
    Responsabilités :
    - Chargement robuste du fichier CSV (chemin par défaut ou personnalisé).
    - Validation de la présence des colonnes indispensables.
    - Extraction des variables explicatives (features X) et de la cible (target y).
    - Découpage stratifié en ensembles d'entraînement et de test.
    """

    def __init__(self, path: Optional[str | Path] = None):
        """
        Initialise le dataset en chargeant les données.
        
        Args:
            path: Chemin vers le fichier CSV. Si None, utilise le chemin par défaut
                  data/patient_data.csv relatif à la racine du projet.
        """
        if path is None:
            # Localisation relative robuste du CSV dans l'arborescence
            default_path = Path(__file__).resolve().parents[1] / "data" / "patient_data.csv"
            self.path = default_path
        else:
            self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Fichier de données introuvable : {self.path}")

        self.data: pd.DataFrame = pd.read_csv(self.path)
        self._validate()

    def _validate(self) -> None:
        """Vérifie que le CSV contient bien toutes les colonnes requises."""
        missing_features = [col for col in ALL_FEATURES if col not in self.data.columns]
        if missing_features:
            raise ValueError(f"Colonnes explicatives manquantes dans le CSV : {missing_features}")
        if TARGET_COLUMN not in self.data.columns:
            raise ValueError(f"Colonne cible '{TARGET_COLUMN}' absente du CSV.")

    def features(self) -> pd.DataFrame:
        """
        Retourne une copie du DataFrame contenant uniquement les variables explicatives X.
        """
        return self.data[ALL_FEATURES].copy()

    def target(self) -> pd.Series:
        """
        Retourne une copie de la Series contenant la variable cible y ('infected').
        """
        return self.data[TARGET_COLUMN].copy()

    def split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Sépare le jeu de données en sous-ensembles d'entraînement et de test.
        
        La stratification est activée par défaut en raison du déséquilibre
        de la cible 'infected' (environ 79% sain / 21% infecté).
        
        Args:
            test_size: Proportion de l'échantillon de test (défaut: 0.2 = 20%).
            random_state: Graine pour la reproductibilité.
            stratify: Si True, préserve la distribution de la cible dans train et test.
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = self.features()
        y = self.target()
        stratify_target = y if stratify else None

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target
        )

    def summary(self) -> dict:
        """Retourne un résumé statistique et structurel du dataset."""
        return {
            "total_rows": len(self.data),
            "total_columns": len(self.data.columns),
            "features_count": len(ALL_FEATURES),
            "target_distribution": self.target().value_counts().to_dict(),
            "missing_values": int(self.data.isnull().sum().sum()),
            "duplicates": int(self.data.duplicated().sum())
        }
